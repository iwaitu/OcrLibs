using Emgu.CV;
using Emgu.CV.Structure;
using Microsoft.AspNetCore.Mvc;
using OcrLiteLib;
using System.Drawing;

namespace OcrWebapi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class OcrController : ControllerBase
    {
        private readonly ILogger<OcrController> _logger;
        private OcrLite ocrEngin;

        public OcrController(ILogger<OcrController> logger)
        {
            _logger = logger;
            string appPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelsDir = appPath + "models";
            string detPath = modelsDir + "/ch_ppocr_server_v2.0_det_infer.onnx";
            string clsPath = modelsDir + "/ch_ppocr_mobile_v2.0_cls_infer.onnx";
            string recPath = modelsDir + "/ch_ppocr_server_v2.0_rec_infer.onnx";
            string keysPath = modelsDir + "/ppocr_keys_v1.txt";

            ocrEngin = new OcrLite();
            ocrEngin.InitModels(detPath, clsPath, recPath, keysPath, 4);

        }

        [HttpPost("ocr")]
        public async Task<string> OcrImage(IFormFile file)
        {
            try
            {
                using var ms = new MemoryStream();
                await file.CopyToAsync(ms);
                using Bitmap bmp = new Bitmap(ms);
                Image<Bgr, byte> src = bmp.ToImage<Bgr, byte>();
                var ret = ocrEngin.Detect(src.Mat);
                return ret.StrRes;
            }
            catch (Exception ex)
            {

                _logger.LogError(ex.Message);
                return "error ";
            }
           
        }

    }
}