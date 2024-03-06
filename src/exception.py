import sys

def error_massage_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_name
    error_message = f"Error ocurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = error_massage_detail(error_message,error_detail)

    def __str__(self) -> str:
        return self.error_message
    