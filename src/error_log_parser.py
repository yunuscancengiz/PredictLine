import pandas as pd
import os


class ErrorLogParser:
    def __init__(self, log_file:str) -> None:
        self.path = os.path.join(os.getcwd(), '..', 'error report')
        self.log_file = self.path + '/' + log_file
        self.list_for_df = []
        self.empty_info = {'installed_machine': '', 'removed_machine': '', 'msg1': '', 'msg2': '', 'msg3': '', 'time': ''}

    def parse(self):
        with open(self.log_file, 'r', encoding='utf-8') as f:
            error_info = self.empty_info
            counter = 0
            for line in f:
                if line == '\n':
                    print(error_info)
                    self.list_for_df.append(error_info)
                    error_info = self.empty_info
                    counter = 0
                else:
                    if counter == 0:
                        error_info['installed_machine'] = line.rstrip('\n').lstrip('#')
                    elif counter == 1:
                        error_info['removed_machine'] = line.rstrip('\n').lstrip('#')
                    elif counter == 2:
                        error_info['msg1'] = line.rstrip('\n').lstrip('#')
                    elif counter == 3:
                        error_info['msg2'] = line.rstrip('\n').lstrip('#')
                    elif counter == 4:
                        error_info['msg3'] = line.rstrip('\n').lstrip('#')


                    for year in range(2021, 2026):
                        if str(year) in error_info['msg1']:
                            error_info['time'] = error_info['msg1'].strip()[len(error_info['msg1'].split()) - 31:]
                        elif str(year) in error_info['msg2']:
                            error_info['time'] = error_info['msg2'].strip()[len(error_info['msg2'].split()) - 31:]
                        elif str(year) in error_info['msg3']:
                            error_info['time'] = error_info['msg3'].strip()[len(error_info['msg3'].split()) - 31:]
                counter += 1


    def convert_to_excel(self):
        df = pd.DataFrame(self.list_for_df)
        df.to_excel('log_parser_test.xlsx', index=False)


if __name__ == '__main__':
    parser = ErrorLogParser(log_file='error_logs.txt')
    parser.parse()
    parser.convert_to_excel()