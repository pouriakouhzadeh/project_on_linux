# from celery import Celery
# # import tasks

# CELERYD_PREFETCH_MULTIPLIER = 1
# CELERYD_MAX_TASKS_PER_CHILD = 100

# app = Celery('currency_trading_tasks',
#              broker='amqp://pouria:P1755063881k@192.168.12.10',  # این آدرس برای RabbitMQ محلی است. در صورت نیاز تغییر دهید.
#              backend='rpc://',
#              include=['tasks']) 

# # تنظیمات تایم‌اوت
# app.conf.task_time_limit = 86400  # 24 ساعت به ثانیه
# app.conf.task_soft_time_limit = 84600  # 23 ساعت و 30 دقیقه به ثانیه



from celery import Celery
import os

# استفاده از متغیرهای محیطی برای اطلاعات حساس
broker_url = os.getenv('CELERY_BROKER_URL', 'amqp://pouria:P1755063881k@192.168.12.10')
backend_url = os.getenv('CELERY_BACKEND_URL', 'rpc://')

app = Celery('currency_trading_tasks',
             broker=broker_url,
             backend=backend_url,
             include=['tasks'])

# تنظیمات تایم‌اوت و سایر پارامترهای مهم Celery
app.conf.update(
    task_time_limit=86400,  # 24 ساعت به ثانیه
    task_soft_time_limit=84600,  # 23 ساعت و 30 دقیقه به ثانیه
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100
)
