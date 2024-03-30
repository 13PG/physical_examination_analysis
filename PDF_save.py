import fitz # 导入fitz库，但在命令行其实是敲PyMuPDF
import os


# pdf存图片
def pdf_to_images(pdf_path, output_dir):
    index=0
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    file_list = []
    # 遍历PDF中的所有页面
    # for page_num in range(doc.page_count):
    for page_num in range(doc.page_count):
        # 获取页面对象
        page = doc[page_num]
        #增强图片分辨率 (如果这里不进行调整的话，后面识别就会出现pytesseract.pytesseract.TesseractError: (3765269347, '')无法识别的报错)
        zoom_x = 5 #水平方向
        zoom_y = 5 #垂直方向
        # 设置输出图片的矩阵变换，比如缩放比例为1（原比例），无旋转
        mat = fitz.Matrix(zoom_x, zoom_y)

        # 将页面转换为Pixmap对象（像素图）
        pix = page.get_pixmap(matrix=mat, alpha=False) # 若包含透明度，alpha=True

        # 定义图片输出文件名
        img_filename = f"page_{index}.png" # 这里以PNG格式为例
        img_path = os.path.join(output_dir, img_filename)
        file_list.append(img_path)
        # 保存图片到文件
        pix._writeIMG(img_path,jpg_quality=100,format_='PNG')

        index+=1
        # 关闭PDF文档
    doc.close()

    return file_list

# 获取文件夹下文件名
def get_filenames_in_directory(directory_path):
    filenames = []

    # 检查路径是否存在且是个目录
    if os.path.isdir(directory_path):
        # 获取目录下所有文件及文件夹名
        all_items = os.listdir(directory_path)

        # 过滤出仅包含文件的名称
        filenames = [r"C:\Users\Administrator\Desktop\temporary_work\text_recognition\PDFset\\"+f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]

    return filenames

# 使用函数
if __name__ == '__main__':
    index = 0
    directory_to_scan = r"C:\Users\Administrator\Desktop\temporary_work\text_recognition\PDFset"
    # 把文件夹下的文件名全部过滤出来
    file_names = get_filenames_in_directory(directory_to_scan)

    #把对应的pdf文件存储到指定路径下
    for name in file_names:
        pdf_to_images(name, r"C:\Users\Administrator\Desktop\temporary_work\text_recognition\PDF_img")

##特别注意网上很多的fitz方法都是很老的，方法名要改，尤其是保存图片
