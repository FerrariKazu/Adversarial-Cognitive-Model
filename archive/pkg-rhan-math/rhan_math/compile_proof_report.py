#!/usr/bin/env python3
import os
import subprocess
import shutil

def compile_report():
    print("Starting Theoretical Compendium compilation...")
    
    # Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    md_path = os.path.join(base_dir, "temp_proofs.md")
    html_path = os.path.join(base_dir, "temp_proofs.html")
    pdf_path = os.path.join(workspace_dir, "rhan_mathematical_proof.pdf")
    
    # Check what phase markdown files exist
    phases = ["phase1_proofs.md", "phase2_proofs.md", "phase3_proofs.md", "phase4_proofs.md", "phase5_proofs.md"]
    existing_phases = []
    for phase in phases:
        p_path = os.path.join(base_dir, phase)
        if os.path.exists(p_path):
            existing_phases.append(p_path)
            print(f"Found: {phase}")
            
    if not existing_phases:
        print("Error: No phase markdown files found!")
        return
    
    # Injected CSS stylesheet
    # (styles truncated for brevity, but they are defined below in the file)
    # ...
    # Let's keep the rest as is but update the image replacement logic below.


    # Injected CSS stylesheet
    css_content = """<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
  
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #1f2937;
    line-height: 1.6;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 30px;
  }
  
  .title-area {
    text-align: center;
    border-bottom: 3px double #e5e7eb;
    padding-bottom: 30px;
    margin-bottom: 40px;
    margin-top: 100px;
  }
  
  .title-area h1 {
    font-size: 2.8em;
    color: #111827;
    margin-bottom: 10px;
    font-weight: 700;
  }
  
  .title-area p {
    font-size: 1.2em;
    color: #4b5563;
    margin: 5px 0;
  }
  
  h2 {
    font-size: 2.0em;
    color: #1e3a8a; /* Deep blue for main sections */
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
    margin-top: 60px;
    margin-bottom: 25px;
    page-break-before: always;
  }
  
  h3 {
    font-size: 1.4em;
    color: #111827;
    margin-top: 40px;
    margin-bottom: 20px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
  }
  
  h4 {
    font-size: 1.15em;
    color: #374151;
    margin-top: 25px;
    margin-bottom: 10px;
  }
  
  code {
    font-family: 'Fira Code', monospace;
    background-color: #f3f4f6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  
  pre {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 20px 0;
  }
  
  pre code {
    background-color: transparent;
    padding: 0;
    font-size: 0.85em;
  }
  
  /* Callout boxes */
  .callout {
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    padding: 16px 20px;
    margin: 24px 0;
    border-radius: 0 8px 8px 0;
  }
  
  .theorem {
    background-color: #fffbeb;
    border-left: 4px solid #d97706;
    padding: 16px 20px;
    margin: 24px 0;
    border-radius: 0 8px 8px 0;
  }
  
  .proof {
    background-color: #f9fafb;
    border-left: 4px solid #4b5563;
    padding: 16px 20px;
    margin: 24px 0;
    border-radius: 0 8px 8px 0;
  }
  
  /* Figure Styling */
  .figure-container {
    text-align: center;
    margin: 40px auto;
    max-width: 85%;
    page-break-inside: avoid;
  }
  .figure-container img {
    max-width: 100%;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    background-color: #ffffff;
    padding: 8px;
  }
  .figure-caption {
    font-size: 0.9em;
    color: #4b5563;
    margin-top: 12px;
    line-height: 1.5;
  }
</style>
"""

    # Title area
    title_area = """
<div class="title-area">
  <h1>Theoretical Deep Learning Compendium</h1>
  <p><strong>Recurrent Hybrid Attention Network (RHAN-v10)</strong></p>
  <p>Mathematical Proofs, Dimensional Analysis & Biological Foundations</p>
  <p><em>Author: Mina M. Wahib</em></p>
  <p>Adversarial Cognitive Systems Laboratory</p>
  <p>Date: July 2026</p>
</div>
"""

    # Concatenate markdown files
    full_markdown = css_content + title_area + "\n"
    
    for path in existing_phases:
        with open(path, 'r') as f:
            content = f.read()
            # Resolve image paths to absolute file paths for wkhtmltopdf
            assets_dir = os.path.join(base_dir, "assets")
            # Convert relative image paths like src="rank_restoration.png" to absolute path
            content = content.replace('src="rank_restoration.png"', f'src="{os.path.join(assets_dir, "rank_restoration.png")}"')
            content = content.replace('src="gelu_dip.png"', f'src="{os.path.join(assets_dir, "gelu_dip.png")}"')
            content = content.replace('src="spherical_geometry.png"', f'src="{os.path.join(assets_dir, "spherical_geometry.png")}"')
            content = content.replace('src="groupnorm_projection.png"', f'src="{os.path.join(assets_dir, "groupnorm_projection.png")}"')
            content = content.replace('src="gradient_masking_surface.png"', f'src="{os.path.join(assets_dir, "gradient_masking_surface.png")}"')
            content = content.replace('src="hessian_conditioning.png"', f'src="{os.path.join(assets_dir, "hessian_conditioning.png")}"')
            content = content.replace('src="left_null_space.png"', f'src="{os.path.join(assets_dir, "left_null_space.png")}"')
            content = content.replace('src="stn_grid.png"', f'src="{os.path.join(assets_dir, "stn_grid.png")}"')
            content = content.replace('src="precision_gating.png"', f'src="{os.path.join(assets_dir, "precision_gating.png")}"')
            content = content.replace('src="dynamic_trades_gating.png"', f'src="{os.path.join(assets_dir, "dynamic_trades_gating.png")}"')
            content = content.replace('src="banach_contraction_decay.png"', f'src="{os.path.join(assets_dir, "banach_contraction_decay.png")}"')
            content = content.replace('src="deq_memory_complexity.png"', f'src="{os.path.join(assets_dir, "deq_memory_complexity.png")}"')
            content = content.replace('src="act_halting_steps.png"', f'src="{os.path.join(assets_dir, "act_halting_steps.png")}"')
            content = content.replace('src="pgd100_flatline.png"', f'src="{os.path.join(assets_dir, "pgd100_flatline.png")}"')
            content = content.replace('src="specimen_trajectory.png"', f'src="{os.path.join(assets_dir, "specimen_trajectory.png")}"')
            
            full_markdown += content + "\n\n"
            
    with open(md_path, 'w') as f:
        f.write(full_markdown)
        
    print("Temporary markdown written.")
    
    # Convert markdown to HTML via Pandoc
    print("Converting to HTML via Pandoc...")
    subprocess.run([
        "pandoc",
        md_path,
        "-o", html_path,
        "--webtex",
        "--standalone",
        "--metadata", "title=RHAN Theoretical Compendium"
    ], check=True)
    
    # Convert HTML to PDF via wkhtmltopdf
    print("Compiling to PDF via wkhtmltopdf...")
    subprocess.run([
        "wkhtmltopdf",
        "--enable-local-file-access",
        html_path,
        pdf_path
    ], check=True)
    
    # Clean up temp files
    if os.path.exists(md_path):
        os.remove(md_path)
    if os.path.exists(html_path):
        os.remove(html_path)
        
    print(f"Successfully compiled: {pdf_path}")

if __name__ == "__main__":
    compile_report()
