import os
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

# 定义输入文件路径
input_pdf = '/home/liumu/code/光光の任务/PDF拆分重命名/input/12.pdf'

# 创建输出文件夹
output_folder = '/home/liumu/code/光光の任务/PDF拆分重命名/outPut/12/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取输入PDF文件
pdf_reader = PdfReader(input_pdf)

# 获取PDF总页数
total_pages = len(pdf_reader.pages)

# 每4页合并为一个新的PDF文件
file_count = 1
start_page = 0
end_page = 4
while start_page < total_pages:
    # 创建新的PDF写入器
    pdf_writer = PdfWriter()

    # 将每一页添加到新的PDF文件中
    for page_num in range(start_page, min(end_page, total_pages)):
        page = pdf_reader.pages[page_num]
        pdf_writer.add_page(page)

    # 输出新的PDF文件
    output_pdf = f'{output_folder}file_{file_count}.pdf'
    with open(output_pdf, 'wb') as f:
        pdf_writer.write(f)

    # 更新页面范围和文件计数
    start_page = end_page
    end_page += 4
    file_count += 1


print(f'{file_count - 1}个新的PDF文件已生成。')
