from src.utils.io import read_text

def format_options(options):
    return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

def build_prompt(method, row):
    template = read_text(f"prompts/{method}.txt")
    return template.format(
        question=row["question"],
        options=format_options(row["options"])
    )
