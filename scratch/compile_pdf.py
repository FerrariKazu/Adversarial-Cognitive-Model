#!/usr/bin/env python3
import os
import base64
import urllib.request
import subprocess

def main():
    workspace_dir = '/home/ferrarikazu/Adversarial Cognitive Model'
    md_path = os.path.join(workspace_dir, 'RHANarch.md')
    temp_md_path = os.path.join(workspace_dir, 'RHANarch_temp.md')
    img_path = os.path.join(workspace_dir, 'rhan_flowchart.png')
    pdf_path = os.path.join(workspace_dir, 'RHANarch.pdf')

    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    # Read original markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract mermaid block
    start_marker = "```mermaid"
    end_marker = "```"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("No mermaid block found in markdown.")
        return

    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        print("Could not find end of mermaid block.")
        return

    mermaid_code = content[start_idx + len(start_marker):end_idx].strip()

    # Encode to base64 for mermaid.ink API
    # The API expects a UTF-8 base64 string
    encoded_code = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    # Note: URL can sometimes fail if b64 contains special padding or characters, 
    # but standard b64 works perfectly with mermaid.ink
    url = f"https://mermaid.ink/img/{encoded_code}"

    print(f"Downloading rendered flowchart from: {url}")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(img_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Successfully saved diagram image to: {img_path}")
    except Exception as e:
        print(f"Error downloading flowchart from API: {e}")
        return

    # Replace mermaid block with markdown image reference
    new_content = (
        content[:start_idx] +
        f"![RHAN Structural Flowchart]({img_path})" +
        content[end_idx + len(end_marker):]
    )

    with open(temp_md_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("Compiling markdown to PDF using pandoc...")
    compile_cmd = [
        'pandoc',
        temp_md_path,
        '-o', pdf_path,
        '--pdf-engine=pdflatex',
        '-V', 'geometry:margin=1in'
    ]
    try:
        subprocess.run(compile_cmd, check=True)
        print(f"Successfully compiled PDF to: {pdf_path}")
    except Exception as e:
        print(f"Pandoc compilation failed: {e}")
    finally:
        # Clean up temporary markdown
        if os.path.exists(temp_md_path):
            os.remove(temp_md_path)

if __name__ == '__main__':
    main()
