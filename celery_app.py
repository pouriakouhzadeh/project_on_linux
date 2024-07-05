from celery import Celery
import os

# Use environment variables for sensitive information
broker_url = os.getenv('CELERY_BROKER_URL', 'amqp://pouria:P1755063881k@192.168.12.10')
backend_url = os.getenv('CELERY_BACKEND_URL', 'rpc://')

app = Celery('currency_trading_tasks',
             broker=broker_url,
             backend=backend_url,
             include=['tasks'])

app.conf.update(
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_connect_timeout': 10,
        'socket_keepalive': True,
        'socket_timeout': 10
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
