"""Approximate word count of a LaTeX manuscript (text words; displayed/inline math, commands, comments and
float contents other than captions removed).  Usage: python texcount.py main.tex  -> per-file and total counts."""
import os
import re
import sys


def expand(path, seen=None):
    base = os.path.dirname(path)
    txt = open(path, encoding="utf8").read()
    txt = re.sub(r"(?<!\\)%.*", "", txt)
    def rep(m):
        f = m.group(1)
        f = f if f.endswith(".tex") else f + ".tex"
        p = os.path.join(base, f)
        return expand(p) if os.path.exists(p) else ""
    return re.sub(r"\\input\{([^}]*)\}", rep, txt)


def words(txt):
    txt = re.sub(r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}", " ", txt, flags=re.S)
    txt = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", txt, flags=re.S)
    txt = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", txt, flags=re.S)
    txt = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$[^$]*\$", " X ", txt, flags=re.S)     # math -> one token
    txt = re.sub(r"\\(cite[pt]?|ref|eqref|label|includegraphics|bibliography\w*|input|usepackage|documentclass)"
                 r"(\[[^\]]*\])*\{[^}]*\}", " ", txt)
    txt = re.sub(r"\\[a-zA-Z@]+\*?", " ", txt)
    txt = re.sub(r"[{}\[\]&~\\]", " ", txt)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.,;:()/]*", txt))


if __name__ == "__main__":
    main = sys.argv[1]
    full = expand(main)
    body = full.split(r"\begin{document}")[1] if r"\begin{document}" in full else full
    body = body.split(r"\bibliographystyle")[0]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    print(f"abstract {words(abstract.group(1)) if abstract else 0}")
    print(f"total (abstract to end of back matter, captions included) {words(body)}")
