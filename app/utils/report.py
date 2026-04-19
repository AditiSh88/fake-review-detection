from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(text, prediction, confidence):
    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Review: {text}", styles['Normal']))
    content.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence}", styles['Normal']))

    doc.build(content)

    return file_path
