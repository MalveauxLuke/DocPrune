#!/usr/bin/env bash
# Shared fail-closed Poppler preflight for every M3DocVQA launcher.

: "${PDFTOOLS_DIR:?set PDFTOOLS_DIR to the pinned Poppler environment}"
: "${ENV_DIR:?set ENV_DIR to the validated docprune-sol environment}"

test -d "$PDFTOOLS_DIR"
test ! -L "$PDFTOOLS_DIR"
PDFINFO="$PDFTOOLS_DIR/bin/pdfinfo"
PDFTOPPM="$PDFTOOLS_DIR/bin/pdftoppm"
test -f "$PDFINFO"
test ! -L "$PDFINFO"
test -x "$PDFINFO"
test -f "$PDFTOPPM"
test ! -L "$PDFTOPPM"
test -x "$PDFTOPPM"

PDFINFO_SHA256="$(sha256sum "$PDFINFO" | awk '{print $1}')"
PDFTOPPM_SHA256="$(sha256sum "$PDFTOPPM" | awk '{print $1}')"
test "$PDFINFO_SHA256" = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
test "$PDFTOPPM_SHA256" = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"
test "$("$PDFINFO" -v 2>&1 | sed -n '1p')" = "pdfinfo version 26.05.0"
test "$("$PDFTOPPM" -v 2>&1 | sed -n '1p')" = "pdftoppm version 26.05.0"

# pdf2image resolves pdftoppm through PATH; keep the pinned directory ahead
# of the model environment and any host-provided Poppler installation.
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
