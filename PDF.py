# 一、只适合读取pdf为一栏的情况，两栏只会读取左边一栏
# import pdfplumber
# file_path = r'/Users/tjt/Desktop/test.pdf'
# with pdfplumber.open(file_path) as pdf:
#     page = pdf.pages[1]
#     print(page.extract_text())

# 实现两栏格式的pdf文件内容的读取（存在的问题：句子仍旧按照原始排版进行换行）
import sys
import importlib
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams,LTItem,LTCurve,LTFigure
from pdfminer.pdfpage import PDFPage

# 输入需要读取内容的pdf文件路径
path = r'/Users/tjt/Desktop/关于图卷积网络的论文/1.pdf'

def parse():
    fp = open(path, 'rb')
    praser = PDFParser(fp)
    doc = PDFDocument(praser)

    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr,laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr,device)

        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()
            for x in layout:
                if isinstance(x,LTTextBoxHorizontal):
                    with open(r'/Users/tjt/Desktop/关于图卷积网络的论文/1', 'a', encoding='utf-8') as f:
                        results = x.get_text()
                        f.write(results + '\n')

if __name__ == '__main__':
    parse()