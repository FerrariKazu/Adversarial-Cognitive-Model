import re

def main():
    file_path = 'Paper/ACD_paper_v1.tex'
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the bounds of the longtable
    start_marker = '\\begin{longtable}'
    end_marker = '\\end{longtable}'
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find longtable in paper!")
        return

    table_content = content[start_idx:end_idx]

    # Pattern to match \texttt{...} inside the table
    # We use a pattern that matches \texttt{something}
    pattern = r'\\texttt\{([^{}]+)\}'
    
    def replace_fn(match):
        filename = match.group(1)
        # Replace \_ with \_\allowbreak{}
        filename = filename.replace('\\_', '\\_\\allowbreak{}')
        # Replace - with -\allowbreak{}
        filename = filename.replace('-', '-\\allowbreak{}')
        return f'\\texttt{{{filename}}}'

    new_table_content = re.sub(pattern, replace_fn, table_content)
    
    # Reassemble the file content
    new_content = content[:start_idx] + new_table_content + content[end_idx:]

    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully processed filenames in longtable!")

if __name__ == '__main__':
    main()
