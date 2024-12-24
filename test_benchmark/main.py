import os
import subprocess

def run_tests_in_directory(directory):
    # 遍历指定目录及其子目录，找到所有符合格式 test_*.py 的文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                # 构建测试模块名称（不包含.py扩展名）
                test_module_name = file[:-3]  # 去掉.py扩展名
                # 构建完整的文件路径
                test_file_path = os.path.join(root, file)
                # 构建输出文件的路径
                output_file_path = os.path.join(root, file + '.txt')
                # 执行测试命令
                try:
                    if test_module_name == 'test_window':
                        print(file)
                    # with open('test_list.txt', 'a') as f:
                    #     f.write(f"{test_module_name}\n")



                    # print(test_module_name)
                    # result = subprocess.run(
                    #     ['python', '-m', 'test', test_module_name],
                    #     stdout=subprocess.PIPE,
                    #     stderr=subprocess.PIPE,
                    #     text=True,
                    #     cwd=root  # 指定当前工作目录为文件所在的目录
                    # )

                    # if 'SUCCESS' not in str(result):
                    #     print("fail:", test_module_name)
                    # else: 
                    #     print("success: ", test_module_name)

                    # 将结果写入文件
                    # with open(output_file_path, 'w') as output_file:
                    #     output_file.write(result.stdout)
                    #     output_file.write(result.stderr)
                    # print(f'Results for {test_module_name} written to {output_file_path}')
                except subprocess.CalledProcessError as e:
                    print(f'An error occurred while running tests for {test_module_name}: {e}')

# 替换为你的测试目录路径
test_directory = '/root/asterinas/test/build/initramfs/usr/lib/python3.10'
# test_directory = '/usr/lib/python3.10/test'
# run_tests_in_directory(test_directory)



# result = subprocess.run(
#                         ['python', '-m', 'test', 'test_builtin'],
#                         stdout=subprocess.PIPE,
#                         stderr=subprocess.PIPE,
#                         text=True,
#                     )


# print('\n', result, '\n')

# if 'SUCCESS' in str(result):
#     print("good")

with open('test_list.txt') as file:
    for line in file:
        # 
        test_module_name = line.strip()
        print(test_module_name, end=' ')
        
        # run the test by subprocess command
        result = subprocess.run(
            ['python', '-m', 'test', test_module_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # check whether result is success
        if 'SUCCESS' not in str(result):
            print("fail")
            # print(str(result))
        else: 
            print("success")