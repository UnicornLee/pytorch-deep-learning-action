# coding: utf-8
from PIL import Image

def modify_image_bit_depth(image_path):
    """将一张深度为32位的图片修改为24位，通常意味着从32位RGBA（每个通道8位，包括红、绿、蓝和透明度Alpha通道）
    转换为24位RGB（每个通道8位，仅红、绿、蓝，没有透明度）。
    1. Alpha通道处理：32位通常包含透明度（Alpha），转换为24位会丢弃这一信息，背景可能会变为默认颜色（如白色或黑色）。
    2. 文件格式：
        PNG支持32位（RGBA）和24位（RGB）
        JPEG只支持24位RGB
    :param image_path:
    :return:
    """
    # 打开 32 位图片
    image = Image.open(image_path)  # 假设图片是 RGBA 格式

    # 检查当前模式
    print(image.mode)  # 如果输出 'RGBA'，则为 32 位

    # 转换为 24 位 RGB（丢弃 Alpha 通道）
    image_24bit = image.convert('RGB')

    # 保存新图片
    image_24bit.save(image_path.split('.')[0] + ".jpg")  # JPEG 不支持 Alpha，默认保存为 24 位


if __name__ == "__main__":
    modify_image_bit_depth("D:/Code/PythonCode/pytorch-deep-learning-action/data/1/dog1.png")