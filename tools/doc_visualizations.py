import json
from pathlib import Path

RESULTS_FILE = Path("./results/results.json")
DOCS_SAVE_PATH = Path("./docs/vis")
VIS_BASE_URL = (
    "https://github.com/AidinHamedi/ML-Optimizer-Benchmark/raw/vis-ref/results"
)
FILE_FORMAT = ".jpg"

HTML_TEMPLATE = """<!-- This file was auto-generated. Do not edit manually. -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title} Results</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #f5f5f5;
      color: #222;
      margin: 0;
      padding: 2rem;
    }}
    h1 {{
      text-align: center;
      margin-bottom: 3.5rem;
      font-size: 2rem;
      font-weight: 600;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.5rem;
    }}
    @media (max-width: 1024px) {{
      .gallery {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}
    @media (max-width: 640px) {{
      .gallery {{
        grid-template-columns: 1fr;
      }}
    }}
    .card {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    .image-wrapper {{
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #eee;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .image-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .fallback {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      background: #f0f0f0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
      font-weight: 500;
      color: #888;
      text-align: center;
    }}
    .enlarge-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(255,255,255,0.8);
      border: none;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 18px;
      transition: background 0.2s ease;
    }}
    .enlarge-btn:hover {{
      background: rgba(255,255,255,1);
    }}
    .card p {{
      margin: 0;
      padding: 0.75rem;
      font-weight: 500;
      font-size: 0.95rem;
      color: #333;
      background: #fafafa;
      border-top: 1px solid #eee;
    }}
    .modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      justify-content: center;
      align-items: center;
      z-index: 1000;
    }}
    .modal.active {{
      display: flex;
    }}
    .modal-content {{
      position: relative;
      background: white;
      padding: 1rem;
      border-radius: 8px;
      max-width: 90%;
      max-height: 90%;
    }}
    .modal-content img {{
      max-width: 100%;
      max-height: 80vh;
      display: block;
    }}
    .close-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(0,0,0,0.6);
      color: white;
      border: none;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      font-size: 18px;
      cursor: pointer;
    }}
    .close-btn:hover {{
      background: rgba(0,0,0,0.8);
    }}
    footer {{
      margin-top: 4rem;
      text-align: center;
      font-size: 0.9rem;
      color: #555;
    }}
    footer a {{
      color: #0077cc;
      text-decoration: none;
    }}
    footer a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <h1>{title} Benchmark Results</h1>

  <div class="gallery">
    {cards}
  </div>

  <div class="modal" id="imageModal">
    <div class="modal-content">
      <button class="close-btn" onclick="closeModal()">×</button>
      <img id="modalImage" src="" alt="Enlarged view">
    </div>
  </div>

  <footer>
    Original repo: <a href="https://github.com/AidinHamedi/ML-Optimizer-Benchmark" target="_blank">github.com/AidinHamedi/ML-Optimizer-Benchmark</a><br>
    Made with ❤️ by AidinHamedi<br>
    <em>This page was automatically generated. Changes may be overwritten.</em>
  </footer>

  <script>
    function openModal(src) {{
      const modal = document.getElementById('imageModal');
      const modalImg = document.getElementById('modalImage');
      modalImg.src = src;
      modal.classList.add('active');
    }}
    function closeModal() {{
      const modal = document.getElementById('imageModal');
      modal.classList.remove('active');
    }}
  </script>
</body>
</html>
"""

CARD_TEMPLATE = """
<div class="card">
  <div class="image-wrapper">
    <img src="{url}" alt="{name}"
         onerror="this.parentNode.innerHTML='<div class=&quot;fallback&quot;>Not available</div>'">
    <button class="enlarge-btn" onclick="openModal('{url}')">🔍</button>
  </div>
  <p>{name}</p>
</div>
"""


def main():
    print("Reading results...")
    if RESULTS_FILE.exists():
        try:
            with RESULTS_FILE.open("r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            raise Exception(f"Error occurred while opening the results file: {e}")
    else:
        raise FileNotFoundError(f"Results file not found at {RESULTS_FILE}")

    print("Generating visualization docs...")
    for optimizer in results["optimizers"]:
        print(f" - Generating visualizations for {optimizer}...")
        cards_html = ""
        for function in results["optimizers"][optimizer]["error_rates"]:
            path = (
                VIS_BASE_URL + "/" + optimizer + "/" + function + FILE_FORMAT
            ).replace(" ", "%20")
            print(f"    - {function} (url: {path})")
            cards_html += CARD_TEMPLATE.format(url=path, name=function)

        html_content = HTML_TEMPLATE.format(title=optimizer, cards=cards_html)

        DOCS_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        DOCS_SAVE_PATH.joinpath(f"{optimizer}.html").write_text(
            html_content, encoding="utf-8"
        )

    print("Done.")


if __name__ == "__main__":
    main()
