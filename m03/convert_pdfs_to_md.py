import os
import pypandoc

def convert_pdf_to_md(pdf_path, output_dir):
    """Convert a PDF file to Markdown format."""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename with .md extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.md")
    
    try:
        # Convert PDF to Markdown
        output = pypandoc.convert_file(
            pdf_path, 
            'md', 
            outputfile=output_path,
            extra_args=['--wrap=none']  # Don't wrap lines
        )
        print(f"Successfully converted: {pdf_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return False

def main():
    # Directory containing PDFs
    pdf_dir = r"D:\project\m03"
    # Output directory for Markdown files
    output_dir = os.path.join(pdf_dir, "markdown")
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert...")
    
    # Convert each PDF to Markdown
    success_count = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if convert_pdf_to_md(pdf_path, output_dir):
            success_count += 1
    
    print(f"\nConversion complete. Successfully converted {success_count} out of {len(pdf_files)} files.")
    print(f"Markdown files saved in: {output_dir}")

if __name__ == "__main__":
    main()
