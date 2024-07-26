


if __name__ == '__main__':
    file_path = "./torch_cambrian_keys"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()  # 读取整个文件内容

    print(f"len: {len(content)}")

    sorted_content = sorted(content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(sorted_content)
