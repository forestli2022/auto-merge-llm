from loguru import logger
from datetime import datetime
import sys


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"output_{timestamp}.log"
logger.remove()  
logger.add(sys.stdout, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}")
logger.add(log_filename, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}")