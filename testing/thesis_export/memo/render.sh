#!/bin/bash
# Build the blind-validation memo: figures -> PDF -> one PNG per page.
#
#   bash testing/thesis_export/memo/render.sh
#
# Source lives here and is committed (memo.html); everything it produces
# lands in testing/thesis_export/generated/memo/, which is gitignored —
# same "commit the generator, not the output" rule as the rest of testing/.
#
# The PDF and the page images are rendered from the same HTML, so they can
# never disagree with each other. Figures are regenerated from the live
# pipeline first, so neither can disagree with the results tables either.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC_DIR="$REPO_ROOT/testing/thesis_export/memo"
OUT_DIR="$REPO_ROOT/testing/thesis_export/generated/memo"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [ ! -x "$CHROME" ]; then
  echo "error: headless Chrome not found at $CHROME (needed for HTML->PDF)" >&2
  exit 1
fi

cd "$REPO_ROOT"
# absolute: the rasterizing steps below run from $OUT_DIR, not the repo root
PY="${PY:-$REPO_ROOT/.venv/bin/python3}"

echo "1/4  regenerating figures from the current pipeline"
"$PY" -m testing.thesis_export.build_memo_figures >/dev/null

mkdir -p "$OUT_DIR"
cp "$SRC_DIR/memo.html" "$OUT_DIR/memo.html"
cd "$OUT_DIR"

echo "2/4  rendering memo.pdf"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="memo.pdf" --virtual-time-budget=8000 \
  "file://$OUT_DIR/memo.html" >/dev/null 2>&1

echo "3/4  splitting pages"
rm -f _page*.html memo_p*.png
"$PY" - <<'PY'
html = open('memo.html').read()
head = html.split('<body>')[0] + '<body>\n'
# Split structurally on the page marker rather than on comment delimiters:
# each chunk's own content ends at its last </div>, and anything after that
# is just whitespace/comments before the next page.
chunks = html.split('<body>', 1)[1].split('<div class="page">')[1:]
for i, chunk in enumerate(chunks, 1):
    end = chunk.rfind('</div>') + len('</div>')
    page = '<div class="page">' + chunk[:end]
    doc = head.replace('</style>', '  body{margin:0}\n  .page{page-break-after:auto}\n</style>') + page + '\n</body></html>'
    open('_page%d.html' % i, 'w').write(doc)
print(len(chunks))
PY

echo "4/4  rasterizing pages at 1224x1584 (144 DPI)"
N=$(ls _page*.html | wc -l | tr -d ' ')
for i in $(seq 1 "$N"); do
  "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=1.5 \
    --window-size=816,1056 --screenshot="memo_p${i}.png" --virtual-time-budget=6000 \
    "file://$OUT_DIR/_page${i}.html" >/dev/null 2>&1
done
rm -f _page*.html

echo
echo "done -> $OUT_DIR"
"$PY" -c "
from PIL import Image; import glob
for f in sorted(glob.glob('memo_p*.png')):
    print('  ', f, Image.open(f).size)"
