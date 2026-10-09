"""Standard-library citation validator; runs inside the sandbox."""
import json
import re
import sys
REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

def check(report_text, sources):
    report_text = report_text.replace("\r\n", "\n").replace("\r", "\n")
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected nonempty array)"]
    numbers, urls = {}, set()
    for source in sources:
        if not isinstance(source, dict):
            problems.append("source must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int or n < 1:
            problems.append("source n must be a positive integer")
            continue
        if n in numbers:
            problems.append(f"duplicate source number [{n}]")
        numbers[n] = source
        family, identifier = source.get("source"), source.get("id")
        if family in {"hf-daily", "hf-search"}:
            if not isinstance(identifier, str) or not identifier or url != "https://huggingface.co/papers/" + identifier:
                problems.append(f"source [{n}] invalid Hugging Face source URL/id: id={identifier!r}, url={url!r}; copy the exact id/url pair from researcher notes")
        elif family == "arxiv":
            if not isinstance(identifier, str) or not identifier or re.search(r"v\d+$", identifier) or url != "https://arxiv.org/abs/" + identifier:
                problems.append(f"source [{n}] invalid arxiv source URL/id: id={identifier!r}, url={url!r}")
        elif family is not None and family != "web":
            problems.append(f"source [{n}] invalid source label: {family!r}")
        if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
            problems.append(f"invalid URL for [{n}]")
        elif url in urls:
            problems.append(f"duplicate URL for [{n}]")
        else:
            urls.add(url)
    clean = re.sub(r"^\x60{3,}.*?^\x60{3,}[^\n]*$|^~{3,}.*?^~{3,}[^\n]*$", "", report_text, flags=re.M | re.S)
    clean = re.sub(r"\x60[^\x60\n]*\x60", "", clean)
    heading = re.search(r"^## References[ \t]*$", clean, re.M)
    if not heading:
        return problems + ["missing ## References"]
    body, refs = clean[:heading.start()], clean[heading.end():]
    if re.search(r"^\[\d+\].*\b(?:arxiv|hf-daily|hf-search|web)\.\s+https?://", body, re.M):
        problems.append("reference-style rows in report body; keep bibliography only in ## References")
    body = re.sub(r"\[[^\]\n]*\]\([^\n]*?\)", "", body)
    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*(?:,|-)\s*\d+)*)\]", body):
        for part in match.group(1).split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                cited.add(int(bounds[0]))
            else:
                start, end = map(int, bounds)
                if end < start or end - start > 10000:
                    problems.append("invalid citation range")
                else:
                    cited.update(range(start, end + 1))
    for n in sorted(cited - numbers.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(numbers.keys() - cited):
        problems.append(f"source [{n}] never cited")
    seen = set()
    for match in re.finditer(r"^\[(\d+)\]([^\n]*)$", refs, re.M):
        n, line = int(match.group(1)), match.group(2)
        if n in seen:
            problems.append(f"duplicate reference [{n}]")
        seen.add(n)
        if n not in numbers:
            problems.append(f"unknown reference [{n}]")
            continue
        found = re.findall(r"https?://[^\s<>]+", line)
        if len(found) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif found[0] != numbers[n].get("url"):
            problems.append(f"reference [{n}] URL differs from sources.json")
    for n in sorted(numbers.keys() - seen):
        problems.append(f"missing reference [{n}]")
    return problems

def main(argv):
    try:
        with open(argv[1] if len(argv)>1 else REPORT, encoding="utf-8") as f:
            report = f.read()
        with open(argv[2] if len(argv)>2 else SOURCES, encoding="utf-8") as f:
            sources = json.load(f)
        problems = check(report, sources)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
