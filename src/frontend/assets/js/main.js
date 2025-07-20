document.addEventListener('DOMContentLoaded', () => {

  // Select necessary DOM elements
  const form = document.getElementById('upload-form');
  const imageInput = document.getElementById('imageUpload');
  const previewContainer = document.getElementById('preview');
  const previewImage = document.getElementById('preview-img');
  const analysisSection = document.getElementById('analysis-result');
  const resultText = document.getElementById('result-text');
  const downloadButton = document.getElementById('download-report');

  // Show preview of the uploaded image
  imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];

    if (!file) {
      previewContainer.style.display = 'none';
      return;
    }

    // Accept only JPEG and PNG formats
    const validTypes = ['image/jpeg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      alert('Only JPEG and PNG formats are supported.');
      imageInput.value = '';
      previewContainer.style.display = 'none';
      return;
    }

    // Read and display the image
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
  });

  // Handle form submission and display analysis result
  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) {
      alert('Please select a medical image before starting the analysis.');
      return;
    }

    // Simulated analysis result (replace with AI model integration later)
    analysisSection.style.display = 'block';
    resultText.innerText = "Anomali tespit edildi: Sol alt akciğer lobunda gölgelenme.";

    // Make the PDF download button visible
    downloadButton.style.display = 'inline-block';
  });

  // Generate and download PDF from analysis section
  downloadButton.addEventListener('click', async (e) => {
    e.preventDefault();
    const { jsPDF } = window.jspdf;
    const contentElement = document.getElementById('analysis-result');

    // Capture the analysis result section as canvas
    html2canvas(contentElement).then(canvas => {
      const imgData = canvas.toDataURL('image/png');

      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4"
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = (canvas.height * pageWidth) / canvas.width;

      // Add the captured image to the PDF
      pdf.addImage(imgData, 'PNG', 0, 0, pageWidth, pageHeight);

      // Trigger PDF download
      pdf.save("ScanWiseAI_Analysis_Report.pdf");
    });
  });

});
