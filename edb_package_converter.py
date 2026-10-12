#!/usr/bin/env python3
"""
edb_package_converter.py

Conservative EPAS/EDB package -> PostgreSQL Community converter.

What it does
------------
* Finds CREATE [OR REPLACE] PACKAGE and PACKAGE BODY blocks in an SQL file.
* Converts straightforward top-level packaged PROCEDURE/FUNCTION definitions into
  PostgreSQL schema-qualified CREATE OR REPLACE PROCEDURE/FUNCTION statements.
* Maps common Oracle/EPAS signature datatypes to PostgreSQL equivalents.
* Preserves the PL/SQL/SPL declaration and executable sections where they are
  syntactically close to PL/pgSQL.
* Detects risky Oracle/EPAS constructs and writes them to a manual-review report.
* Never silently drops an unconvertible routine.

Important
---------
This is intentionally conservative. PostgreSQL Community has no PACKAGE object,
so package-level state, initialization sections, private types, cursors, nested
subprograms, and some Oracle semantics require design decisions and manual review.

Python: 3.10+
Dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence


IDENT = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$#]*)'
QUALIFIED_IDENT = rf'{IDENT}(?:\s*\.\s*{IDENT})?'

PACKAGE_START_RE = re.compile(
    rf"""
    \bCREATE\s+
    (?:OR\s+REPLACE\s+)?
    PACKAGE\s+
    (?P<body>BODY\s+)?
    (?P<name>{QUALIFIED_IDENT})
    (?=\s|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

SLASH_LINE_RE = re.compile(r"(?m)^[ \t]*/[ \t]*(?:\r?\n|$)")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$#]*")

RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("package-state", re.compile(r"\bPACKAGE\b|\bPACKAGE\s+BODY\b", re.I)),
    ("pragma", re.compile(r"\bPRAGMA\b", re.I)),
    ("autonomous-transaction", re.compile(r"\bAUTONOMOUS_TRANSACTION\b", re.I)),
    ("dbms-package", re.compile(r"\bDBMS_[A-Za-z0-9_$#]*\b", re.I)),
    ("utl-package", re.compile(r"\bUTL_[A-Za-z0-9_$#]*\b", re.I)),
    ("sys-refcursor", re.compile(r"\bSYS_REFCURSOR\b", re.I)),
    ("bulk-collect", re.compile(r"\bBULK\s+COLLECT\b", re.I)),
    ("forall", re.compile(r"\bFORALL\b", re.I)),
    ("percent-type", re.compile(r"%\s*TYPE\b", re.I)),
    ("percent-rowtype", re.compile(r"%\s*ROWTYPE\b", re.I)),
    ("raise-application-error", re.compile(r"\bRAISE_APPLICATION_ERROR\b", re.I)),
    ("sqlcode-sqlerrm", re.compile(r"\bSQLCODE\b|\bSQLERRM\b", re.I)),
    ("sql-cursor-attribute", re.compile(r"\bSQL\s*%\s*(?:ROWCOUNT|FOUND|NOTFOUND|ISOPEN)\b", re.I)),
    ("connect-by", re.compile(r"\bCONNECT\s+BY\b|\bSTART\s+WITH\b", re.I)),
    ("rownum", re.compile(r"\bROWNUM\b", re.I)),
    ("oracle-dual", re.compile(r"\bFROM\s+DUAL\b", re.I)),
    ("nvl-family", re.compile(r"\bNVL2?\s*\(", re.I)),
    ("decode", re.compile(r"\bDECODE\s*\(", re.I)),
    ("sysdate", re.compile(r"\bSYSDATE\b", re.I)),
    ("systimestamp", re.compile(r"\bSYSTIMESTAMP\b", re.I)),
    ("oracle-date-functions", re.compile(r"\b(?:ADD_MONTHS|MONTHS_BETWEEN|LAST_DAY|NEXT_DAY)\s*\(", re.I)),
    ("exception-init", re.compile(r"\bEXCEPTION_INIT\b", re.I)),
    ("edb-spl", re.compile(r"\bSPL\b", re.I)),
)

TYPE_MAP = {
    "NUMBER": "numeric",
    "DEC": "numeric",
    "DECIMAL": "numeric",
    "NUMERIC": "numeric",
    "PLS_INTEGER": "integer",
    "BINARY_INTEGER": "integer",
    "SIMPLE_INTEGER": "integer",
    "INTEGER": "integer",
    "INT": "integer",
    "SMALLINT": "smallint",
    "FLOAT": "double precision",
    "BINARY_FLOAT": "real",
    "BINARY_DOUBLE": "double precision",
    "VARCHAR2": "varchar",
    "NVARCHAR2": "varchar",
    "VARCHAR": "varchar",
    "CHAR": "char",
    "NCHAR": "char",
    "CLOB": "text",
    "NCLOB": "text",
    "BLOB": "bytea",
    "RAW": "bytea",
    "LONG": "text",
    "LONG RAW": "bytea",
    "BOOLEAN": "boolean",
    "DATE": "timestamp without time zone",
    "TIMESTAMP": "timestamp without time zone",
    "SYS_REFCURSOR": "refcursor",
}


@dataclasses.dataclass(frozen=True)
class PackageBlock:
    kind: str  # "spec" or "body"
    full_name: str
    source_schema: str | None
    package_name: str
    sql: str
    content_start: int
    content_end: int


@dataclasses.dataclass(frozen=True)
class Routine:
    package_full_name: str
    package_name: str
    source_schema: str | None
    kind: str  # procedure/function
    name: str
    raw_name: str
    params_text: str
    return_text: str | None
    declaration_text: str
    executable_text: str
    original_sql: str
    start: int
    end: int
    warnings: tuple[str, ...]


@dataclasses.dataclass
class ReportRow:
    package: str
    object_name: str
    object_type: str
    status: str
    target_name: str
    warnings: str
    detail: str = ""


def unquote_ident(identifier: str) -> str:
    identifier = identifier.strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier.lower()


def quote_ident(identifier: str) -> str:
    """Quote only when required, preserving a safe lower-case identifier unquoted."""
    if re.fullmatch(r"[a-z_][a-z0-9_$]*", identifier):
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def split_qualified_name(name: str) -> tuple[str | None, str]:
    parts = [p.strip() for p in re.split(r"\s*\.\s*", name.strip(), maxsplit=1)]
    if len(parts) == 1:
        return None, unquote_ident(parts[0])
    return unquote_ident(parts[0]), unquote_ident(parts[1])


def mask_sql(sql: str) -> str:
    """
    Replace comment/string contents with spaces while preserving offsets/newlines.
    Handles --, /* */, single-quoted literals, double-quoted identifiers and
    PostgreSQL-style dollar quoted strings.
    """
    out = list(sql)
    n = len(sql)
    i = 0

    def blank(a: int, b: int) -> None:
        for j in range(a, b):
            if out[j] not in "\r\n":
                out[j] = " "

    while i < n:
        if sql.startswith("--", i):
            j = sql.find("\n", i + 2)
            if j == -1:
                j = n
            blank(i, j)
            i = j
            continue

        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
            continue

        ch = sql[i]

        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            blank(i, j)
            i = j
            continue

        if ch == '"':
            j = i + 1
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            # Keep identifiers visible to parser, so do not blank them.
            i = j
            continue

        if ch == "$":
            m = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[i:])
            if m:
                tag = m.group(0)
                j = sql.find(tag, i + len(tag))
                j = n if j == -1 else j + len(tag)
                blank(i, j)
                i = j
                continue

        i += 1

    return "".join(out)


def find_matching_paren(masked: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(masked)):
        c = masked[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched '(' at offset {open_pos}")


def next_nonspace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def find_word(masked: str, word: str, start: int, end: int | None = None) -> re.Match[str] | None:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.I)
    return pattern.search(masked, start, len(masked) if end is None else end)


def find_package_content_bounds(block_sql: str, package_name: str) -> tuple[int, int]:
    masked = mask_sql(block_sql)
    start_match = PACKAGE_START_RE.search(masked)
    if not start_match:
        raise ValueError("Not a package block")

    # Find AS/IS after CREATE ... PACKAGE ... name.
    hdr = re.compile(r"\b(?:AUTHID\s+(?:CURRENT_USER|DEFINER)\s+)?(?:AS|IS)\b", re.I)
    hm = hdr.search(masked, start_match.end())
    if not hm:
        raise ValueError("Package AS/IS keyword not found")

    content_start = hm.end()

    # Prefer END <package_name>; at the end of the block.
    name_pat = re.escape(package_name)
    end_named = list(re.finditer(rf"\bEND\s+(?:" + name_pat + r'|"' + name_pat + r'")\s*;', masked, re.I))
    if end_named:
        return content_start, end_named[-1].start()

    # Fallback to final END;
    end_plain = list(re.finditer(r"\bEND\s*;", masked, re.I))
    if end_plain:
        return content_start, end_plain[-1].start()

    raise ValueError("Package closing END not found")


def find_package_blocks(sql: str) -> list[PackageBlock]:
    masked = mask_sql(sql)
    starts = list(PACKAGE_START_RE.finditer(masked))
    blocks: list[PackageBlock] = []

    for idx, match in enumerate(starts):
        start = match.start()
        next_start = starts[idx + 1].start() if idx + 1 < len(starts) else len(sql)

        slash = SLASH_LINE_RE.search(masked, match.end(), next_start)
        end = slash.end() if slash else next_start

        block_sql = sql[start:end]
        source_schema, package_name = split_qualified_name(match.group("name"))
        kind = "body" if match.group("body") else "spec"

        try:
            content_start, content_end = find_package_content_bounds(block_sql, package_name)
        except ValueError:
            # Retain block with whole post-header body for reporting.
            content_start, content_end = 0, len(block_sql)

        blocks.append(
            PackageBlock(
                kind=kind,
                full_name=match.group("name").strip(),
                source_schema=source_schema,
                package_name=package_name,
                sql=block_sql,
                content_start=content_start,
                content_end=content_end,
            )
        )

    return blocks


def split_top_level_commas(text: str) -> list[str]:
    masked = mask_sql(text)
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def map_type(type_text: str) -> tuple[str, list[str]]:
    original = re.sub(r"\s+", " ", type_text.strip())
    warnings: list[str] = []

    if "%" in original:
        warnings.append("signature uses %TYPE/%ROWTYPE")
        return original, warnings

    # Preserve schema-qualified/custom types as written.
    normalized_upper = original.upper()

    # NUMBER(p,s), VARCHAR2(n), RAW(n), TIMESTAMP(p), etc.
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_$#]*(?:\s+[A-Za-z_][A-Za-z0-9_$#]*)?)(\s*\([^)]*\))?", original)
    if m:
        base = re.sub(r"\s+", " ", m.group(1)).upper()
        modifier = m.group(2) or ""

        if base == "NUMBER":
            return f"numeric{modifier}", warnings
        if base in {"VARCHAR2", "NVARCHAR2", "VARCHAR"}:
            return f"varchar{modifier}", warnings
        if base in {"CHAR", "NCHAR"}:
            return f"char{modifier}", warnings
        if base == "RAW":
            warnings.append("RAW length semantics require review")
            return "bytea", warnings
        if base == "TIMESTAMP":
            return f"timestamp{modifier} without time zone", warnings
        if base in TYPE_MAP:
            return TYPE_MAP[base], warnings

    # TIMESTAMP WITH [LOCAL] TIME ZONE
    if re.fullmatch(r"TIMESTAMP(?:\s*\(\d+\))?\s+WITH\s+(?:LOCAL\s+)?TIME\s+ZONE", normalized_upper):
        precision = ""
        pm = re.search(r"\(\d+\)", original)
        if pm:
            precision = pm.group(0)
        warnings.append("TIMESTAMP WITH LOCAL TIME ZONE semantics require review")
        return f"timestamp{precision} with time zone", warnings

    if normalized_upper in TYPE_MAP:
        return TYPE_MAP[normalized_upper], warnings

    # Common Oracle collection/table declarations cannot be signature types directly.
    if re.search(r"\bTABLE\s+OF\b|\bVARRAY\b|\bRECORD\b", normalized_upper):
        warnings.append("collection/record type requires manual mapping")

    return original, warnings


def split_default(expr: str) -> tuple[str, str | None]:
    """Split DEFAULT or := at top level."""
    masked = mask_sql(expr)
    depth = 0
    i = 0
    while i < len(masked):
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0:
            m = re.match(r"\bDEFAULT\b", masked[i:], re.I)
            if m:
                return expr[:i].strip(), expr[i + m.end():].strip()
            if masked.startswith(":=", i):
                return expr[:i].strip(), expr[i + 2:].strip()
        i += 1
    return expr.strip(), None


def convert_parameter(param: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    core, default = split_default(param)
    core = re.sub(r"\bNOCOPY\b", "", core, flags=re.I)
    core = re.sub(r"\s+", " ", core).strip()

    # name [IN|OUT|IN OUT] type
    m = re.match(
        rf"^(?P<name>{IDENT})\s+(?:(?P<mode>IN\s+OUT|INOUT|IN|OUT)\s+)?(?P<type>.+)$",
        core,
        re.I,
    )
    if not m:
        return param.strip(), ["parameter syntax not recognized"]

    name = m.group("name").strip()
    mode = (m.group("mode") or "IN").upper().replace(" ", "")
    type_text = m.group("type").strip()
    mapped_type, type_warnings = map_type(type_text)
    warnings.extend(type_warnings)

    if mode == "IN":
        converted = f"{name} {mapped_type}"
    elif mode == "OUT":
        converted = f"OUT {name} {mapped_type}"
    else:
        converted = f"INOUT {name} {mapped_type}"

    if default is not None:
        if mode == "OUT":
            warnings.append("OUT parameter default is not valid in PostgreSQL")
        converted += f" DEFAULT {default}"

    return converted, warnings


def convert_parameters(params_text: str) -> tuple[str, list[str]]:
    if not params_text.strip():
        return "", []

    converted: list[str] = []
    warnings: list[str] = []
    for param in split_top_level_commas(params_text):
        c, w = convert_parameter(param)
        converted.append(c)
        warnings.extend(w)
    return ", ".join(converted), warnings


def detect_risks(text: str) -> list[str]:
    found = []
    for label, pattern in RISK_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return sorted(set(found))


def find_keyword_outside_parens(masked: str, words: Sequence[str], start: int) -> tuple[str, int, int] | None:
    depth = 0
    word_set = {w.upper() for w in words}
    i = start
    while i < len(masked):
        ch = masked[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(depth - 1, 0)
            i += 1
            continue
        if depth == 0 and (ch.isalpha() or ch == "_"):
            m = WORD_RE.match(masked, i)
            if m:
                word = m.group(0).upper()
                if word in word_set:
                    return word, m.start(), m.end()
                i = m.end()
                continue
        i += 1
    return None


def find_semicolon(masked: str, start: int) -> int | None:
    depth = 0
    for i in range(start, len(masked)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth = max(depth - 1, 0)
        elif masked[i] == ";" and depth == 0:
            return i
    return None


def parse_routine_header(content: str, start: int) -> dict[str, object] | None:
    masked = mask_sql(content)
    km = re.match(r"\b(PROCEDURE|FUNCTION)\b", masked[start:], re.I)
    if not km:
        return None

    kind = km.group(1).lower()
    pos = start + km.end()
    pos = next_nonspace(masked, pos)

    nm = re.match(IDENT, content[pos:], re.I)
    if not nm:
        return None
    raw_name = nm.group(0)
    name = unquote_ident(raw_name)
    pos += nm.end()
    pos = next_nonspace(masked, pos)

    params_text = ""
    if pos < len(masked) and masked[pos] == "(":
        close = find_matching_paren(masked, pos)
        params_text = content[pos + 1:close]
        pos = next_nonspace(masked, close + 1)

    return_text = None
    if kind == "function":
        ret = find_keyword_outside_parens(masked, ["RETURN"], pos)
        if not ret:
            return None
        _, rstart, rend = ret
        if find_semicolon(masked, pos) is not None and find_semicolon(masked, pos) < rstart:
            return {"declaration_only": True, "end": find_semicolon(masked, pos) + 1}
        body_kw = find_keyword_outside_parens(masked, ["IS", "AS"], rend)
        if not body_kw:
            return None
        _, bstart, bend = body_kw
        semi = find_semicolon(masked, rend)
        if semi is not None and semi < bstart:
            return {"declaration_only": True, "end": semi + 1}
        return_text = content[rend:bstart].strip()
        is_as_start, is_as_end = bstart, bend
    else:
        body_kw = find_keyword_outside_parens(masked, ["IS", "AS"], pos)
        semi = find_semicolon(masked, pos)
        if body_kw is None or (semi is not None and semi < body_kw[1]):
            return {"declaration_only": True, "end": (semi + 1 if semi is not None else pos)}
        _, is_as_start, is_as_end = body_kw

    return {
        "declaration_only": False,
        "kind": kind,
        "raw_name": raw_name,
        "name": name,
        "params_text": params_text,
        "return_text": return_text,
        "is_as_start": is_as_start,
        "is_as_end": is_as_end,
    }


def find_named_routine_end(masked: str, name: str, start: int) -> tuple[int, int] | None:
    escaped = re.escape(name)
    pattern = re.compile(rf'\bEND\s+(?:"{escaped}"|{escaped})\s*;', re.I)
    m = pattern.search(masked, start)
    if m:
        return m.start(), m.end()
    return None


def find_structural_routine_end(masked: str, begin_pos: int) -> tuple[int, int] | None:
    """
    Match BEGIN/END, IF/END IF, CASE/END CASE, LOOP/END LOOP.
    Assumes begin_pos points at the main BEGIN of a routine.
    """
    token_re = re.compile(r"\b(BEGIN|IF|CASE|LOOP|END)\b", re.I)
    tokens = list(token_re.finditer(masked, begin_pos))
    stack: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        word = tok.group(1).upper()

        if word in {"BEGIN", "IF", "CASE", "LOOP"}:
            stack.append(word)
            i += 1
            continue

        # END
        j = next_nonspace(masked, tok.end())
        following = WORD_RE.match(masked, j)
        closing_kind = None
        if following:
            fw = following.group(0).upper()
            if fw in {"IF", "CASE", "LOOP"}:
                closing_kind = fw

        if not stack:
            return None

        if closing_kind:
            # Pop the nearest matching construct; if malformed, fail conservatively.
            if stack[-1] != closing_kind:
                return None
            stack.pop()
            # The IF/CASE/LOOP immediately following END is also present in the
            # token stream. Skip it so it is not mistaken for a new opener.
            if i + 1 < len(tokens) and tokens[i + 1].start() == following.start():
                i += 1
        else:
            if stack[-1] != "BEGIN":
                return None
            stack.pop()

        if not stack:
            semi = masked.find(";", tok.end())
            if semi == -1:
                return None
            return tok.start(), semi + 1

        i += 1

    return None


def extract_routines(block: PackageBlock) -> tuple[list[Routine], list[str]]:
    content = block.sql[block.content_start:block.content_end]
    masked = mask_sql(content)
    routines: list[Routine] = []
    package_warnings: list[str] = []
    pos = 0

    while True:
        m = re.search(r"\b(PROCEDURE|FUNCTION)\b", masked[pos:], re.I)
        if not m:
            break
        start = pos + m.start()
        header = parse_routine_header(content, start)
        if not header:
            package_warnings.append(f"Could not parse routine near offset {start}")
            pos = start + len(m.group(0))
            continue

        if bool(header["declaration_only"]):
            pos = int(header["end"])
            continue

        name = str(header["name"])
        raw_name = str(header["raw_name"])
        kind = str(header["kind"])
        params_text = str(header["params_text"])
        return_text = header["return_text"]
        is_as_end = int(header["is_as_end"])

        # A local routine before the first BEGIN makes structural parsing ambiguous.
        first_begin = find_word(masked, "BEGIN", is_as_end)
        if not first_begin:
            package_warnings.append(f"{kind} {name}: BEGIN not found")
            pos = is_as_end
            continue

        local_subprogram = re.search(r"\b(PROCEDURE|FUNCTION)\b", masked[is_as_end:first_begin.start()], re.I)
        warnings: list[str] = []
        if local_subprogram:
            warnings.append("nested/local subprogram in declaration section")

        # Prefer END routine_name; because it survives nested BEGIN/END structures well.
        end_match = find_named_routine_end(masked, name, is_as_end)
        if end_match:
            end_start, end_end = end_match
        else:
            structural = find_structural_routine_end(masked, first_begin.start())
            if not structural:
                package_warnings.append(f"{kind} {name}: could not locate routine END")
                pos = is_as_end
                continue
            end_start, end_end = structural

        # If a named end was used, determine the main BEGIN. When there are local
        # subprograms this first BEGIN may belong to the local routine, so flag it.
        main_begin = first_begin.start()

        decl_text = content[is_as_end:main_begin].strip()
        executable_text = content[first_begin.end():end_start].strip()
        original_sql = content[start:end_end].strip()

        warnings.extend(detect_risks(original_sql))
        if re.search(r"\bPIPELINED\b|\bRESULT_CACHE\b|\bPARALLEL_ENABLE\b", original_sql, re.I):
            warnings.append("advanced function clause requires manual review")

        routines.append(
            Routine(
                package_full_name=block.full_name,
                package_name=block.package_name,
                source_schema=block.source_schema,
                kind=kind,
                name=name,
                raw_name=raw_name,
                params_text=params_text,
                return_text=str(return_text) if return_text is not None else None,
                declaration_text=decl_text,
                executable_text=executable_text,
                original_sql=original_sql,
                start=start,
                end=end_end,
                warnings=tuple(sorted(set(warnings))),
            )
        )
        pos = end_end

    # Look for package-level initialization after removing routine spans.
    residual = list(content)
    for routine in routines:
        for i in range(routine.start, min(routine.end, len(residual))):
            if residual[i] not in "\r\n":
                residual[i] = " "
    residual_text = "".join(residual)

    residual_masked = mask_sql(residual_text)
    init_match = re.search(r"\bBEGIN\b", residual_masked, re.I)
    if init_match:
        package_warnings.append("package initialization section detected")
    if re.search(r"\b(?:TYPE|SUBTYPE|CURSOR|PRAGMA)\b", residual_masked, re.I):
        package_warnings.append("package-level TYPE/SUBTYPE/CURSOR/PRAGMA detected")

    # Anything resembling a package-level declaration (for example
    # g_counter NUMBER := 0;) represents package state or a forward declaration
    # that Community PostgreSQL cannot reproduce as a PACKAGE variable.
    declaration_zone = residual_masked[: init_match.start()] if init_match else residual_masked
    declaration_zone = re.sub(r"\bEND\b(?:\s+" + IDENT + r")?\s*;", " ", declaration_zone, flags=re.I)
    if re.search(r"[^;\s]+[^;]*;", declaration_zone, re.S):
        package_warnings.append("package-level declarations/state detected")

    return routines, sorted(set(package_warnings))


def normalize_declaration_section(decl: str) -> str:
    """
    Keep declarations mostly intact. Remove a leading DECLARE if an EPAS export
    contains one, because the generator adds DECLARE itself.
    """
    decl = decl.strip()
    decl = re.sub(r"^\s*DECLARE\b", "", decl, flags=re.I).strip()
    return decl


def make_target_schema(block_or_routine: PackageBlock | Routine, mode: str) -> str:
    pkg = block_or_routine.package_name
    src = block_or_routine.source_schema

    if mode == "package":
        return pkg
    if mode == "source_package":
        return f"{src}_{pkg}" if src else pkg
    raise ValueError(f"Unknown schema mode: {mode}")


def convert_routine(routine: Routine, schema_mode: str) -> tuple[str | None, list[str]]:
    warnings = list(routine.warnings)
    params_sql, param_warnings = convert_parameters(routine.params_text)
    warnings.extend(param_warnings)

    return_sql = None
    if routine.kind == "function":
        if routine.return_text is None:
            warnings.append("function return type missing")
            return None, sorted(set(warnings))
        return_sql, ret_warnings = map_type(routine.return_text)
        warnings.extend(ret_warnings)

    # Nested local subprograms cannot be reliably rewritten by this converter.
    if "nested/local subprogram in declaration section" in warnings:
        return None, sorted(set(warnings))

    target_schema = make_target_schema(routine, schema_mode)
    qschema = quote_ident(target_schema)
    qname = quote_ident(routine.name)

    decl = normalize_declaration_section(routine.declaration_text)
    executable = routine.executable_text.strip()

    # Remove optional END <routine>; from executable if an odd export placed it there.
    executable = re.sub(
        rf"\bEND\s+(?:{re.escape(routine.name)}|{re.escape(routine.raw_name)})\s*;\s*$",
        "",
        executable,
        flags=re.I,
    ).rstrip()

    header = f"CREATE OR REPLACE {routine.kind.upper()} {qschema}.{qname}({params_sql})"
    if routine.kind == "function":
        header += f"\nRETURNS {return_sql}"

    lines = [
        header,
        "LANGUAGE plpgsql",
        "AS $edb_pkg$",
    ]
    if decl:
        lines.append("DECLARE")
        lines.append(decl)
    lines.append("BEGIN")
    if executable:
        lines.append(executable)
    lines.append("END;")
    lines.append("$edb_pkg$;")

    return "\n".join(lines), sorted(set(warnings))


def render_manual_block(routine: Routine, warnings: Sequence[str]) -> str:
    warning_text = ", ".join(warnings) if warnings else "manual review required"
    return (
        f"-- ============================================================================\n"
        f"-- MANUAL REVIEW: {routine.package_full_name}.{routine.name}\n"
        f"-- TYPE: {routine.kind}\n"
        f"-- REASONS: {warning_text}\n"
        f"-- ============================================================================\n"
        f"{routine.original_sql}\n"
    )


def convert_file(input_path: Path, output_dir: Path, schema_mode: str) -> int:
    sql = input_path.read_text(encoding="utf-8", errors="replace")
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = find_package_blocks(sql)
    bodies = [b for b in blocks if b.kind == "body"]
    specs = [b for b in blocks if b.kind == "spec"]

    schemas: set[str] = set()
    converted_sql: list[str] = []
    manual_sql: list[str] = []
    report: list[ReportRow] = []

    if not blocks:
        print("ERROR: no CREATE PACKAGE / CREATE PACKAGE BODY blocks found.", file=sys.stderr)
        return 2

    spec_by_pkg = {(b.source_schema, b.package_name): b for b in specs}

    for block in bodies:
        target_schema = make_target_schema(block, schema_mode)
        schemas.add(target_schema)
        routines, package_warnings = extract_routines(block)

        if not routines:
            report.append(
                ReportRow(
                    package=block.full_name,
                    object_name="(package body)",
                    object_type="package",
                    status="MANUAL_REVIEW",
                    target_name=quote_ident(target_schema),
                    warnings="; ".join(package_warnings) or "no routines extracted",
                )
            )
            manual_sql.append(
                f"-- Could not safely extract routines from package body {block.full_name}\n"
                f"{block.sql}\n"
            )
            continue

        if package_warnings:
            report.append(
                ReportRow(
                    package=block.full_name,
                    object_name="(package-level)",
                    object_type="package",
                    status="MANUAL_REVIEW",
                    target_name=quote_ident(target_schema),
                    warnings="; ".join(package_warnings),
                )
            )

        if (block.source_schema, block.package_name) not in spec_by_pkg:
            report.append(
                ReportRow(
                    package=block.full_name,
                    object_name="(package spec)",
                    object_type="package",
                    status="INFO",
                    target_name=quote_ident(target_schema),
                    warnings="package specification not found in input",
                )
            )

        for routine in routines:
            target = f"{quote_ident(target_schema)}.{quote_ident(routine.name)}"
            generated, warnings = convert_routine(routine, schema_mode)

            # Conservative status: risky constructs do not prevent generation, but
            # the report clearly says review is required before deployment.
            blocking = {
                "nested/local subprogram in declaration section",
                "signature uses %TYPE/%ROWTYPE",
                "collection/record type requires manual mapping",
                "parameter syntax not recognized",
            }
            has_blocking = any(w in blocking for w in warnings)

            if generated is None or has_blocking:
                manual_sql.append(render_manual_block(routine, warnings))
                report.append(
                    ReportRow(
                        package=block.full_name,
                        object_name=routine.name,
                        object_type=routine.kind,
                        status="MANUAL_REVIEW",
                        target_name=target,
                        warnings="; ".join(warnings),
                    )
                )
                continue

            converted_sql.append(
                f"-- Source: {block.full_name}.{routine.name}\n"
                + (f"-- REVIEW: {'; '.join(warnings)}\n" if warnings else "")
                + generated
                + "\n"
            )
            report.append(
                ReportRow(
                    package=block.full_name,
                    object_name=routine.name,
                    object_type=routine.kind,
                    status="AUTO_CONVERTED_REVIEW" if warnings else "AUTO_CONVERTED",
                    target_name=target,
                    warnings="; ".join(warnings),
                )
            )

    # Specs without bodies are important.
    body_keys = {(b.source_schema, b.package_name) for b in bodies}
    for spec in specs:
        if (spec.source_schema, spec.package_name) not in body_keys:
            target_schema = make_target_schema(spec, schema_mode)
            schemas.add(target_schema)
            report.append(
                ReportRow(
                    package=spec.full_name,
                    object_name="(package spec)",
                    object_type="package",
                    status="MANUAL_REVIEW",
                    target_name=quote_ident(target_schema),
                    warnings="package body not found in input",
                )
            )
            manual_sql.append(
                f"-- Package specification found without package body: {spec.full_name}\n"
                f"{spec.sql}\n"
            )

    schema_sql = "\n".join(
        f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)};"
        for schema in sorted(schemas)
    ) + ("\n" if schemas else "")

    (output_dir / "00_schemas.sql").write_text(schema_sql, encoding="utf-8")
    (output_dir / "01_routines.sql").write_text(
        "\n\n".join(converted_sql) + ("\n" if converted_sql else ""),
        encoding="utf-8",
    )
    (output_dir / "manual_review.sql").write_text(
        "\n\n".join(manual_sql) + ("\n" if manual_sql else ""),
        encoding="utf-8",
    )

    with (output_dir / "conversion_report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "package",
                "object_name",
                "object_type",
                "status",
                "target_name",
                "warnings",
                "detail",
            ],
        )
        writer.writeheader()
        for row in report:
            writer.writerow(dataclasses.asdict(row))

    readme = f"""EDB package conversion output

Input
-----
{input_path}

Schema mode
-----------
{schema_mode}

Files
-----
00_schemas.sql
    Target schemas corresponding to packages.

01_routines.sql
    Automatically generated PostgreSQL Community routines.
    Any routine carrying a REVIEW comment must be checked before deployment.

manual_review.sql
    Source routines/packages that the converter intentionally did not rewrite.

conversion_report.csv
    Per-package/per-routine conversion status and warnings.

Recommended validation
----------------------
1. Review conversion_report.csv.
2. Review every AUTO_CONVERTED_REVIEW item.
3. Run 00_schemas.sql in a disposable PostgreSQL 18 test database.
4. Run 01_routines.sql with:
       psql -X -v ON_ERROR_STOP=1 -f 01_routines.sql your_database
5. Fix syntax/semantic differences reported by PostgreSQL.
6. Compare routine signatures and application calls.
7. Unit/integration test every converted routine.

Important
---------
Package variables/state, initialization blocks, nested/local subprograms,
package-private types, Oracle collections, autonomous transactions and DBMS_*
semantics require deliberate redesign in PostgreSQL Community.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    auto_count = sum(1 for r in report if r.status.startswith("AUTO_CONVERTED"))
    manual_count = sum(1 for r in report if r.status == "MANUAL_REVIEW")
    print(f"Packages found : {len(blocks)} ({len(specs)} spec, {len(bodies)} body)")
    print(f"Auto converted : {auto_count}")
    print(f"Manual review  : {manual_count}")
    print(f"Output         : {output_dir.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conservatively convert EPAS/EDB package routines to PostgreSQL Community PL/pgSQL."
    )
    parser.add_argument("--input", required=True, type=Path, help="SQL file containing PACKAGE/PACKAGE BODY DDL")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--schema-mode",
        choices=("package", "source_package"),
        default="package",
        help=(
            "package: package PKG_X becomes schema pkg_x, preserving PKG_X.proc() style calls. "
            "source_package: becomes <source_schema>_<package> to reduce cross-schema collisions."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    return convert_file(args.input, args.output, args.schema_mode)


if __name__ == "__main__":
    raise SystemExit(main())
