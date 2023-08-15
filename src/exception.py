import sys 
import logging
from src.logger import logging

def error_message_detail(error, error_detail):
    _, _, tb = sys.exc_info()
    filename=tb.tb_frame.f_code.co_filename
    error_message="Error: {} \nFile: {} \nLine: {} \nDetail: {}".format(error, filename, tb.tb_lineno, error_detail)
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail)
    def __str__(self):
        return self.error_message
    
    
if __name__ == '__main__':
    try:
        a=1/0
    except Exception as e:
        logging.info("devide  by 0")
        raise CustomException(e, sys)