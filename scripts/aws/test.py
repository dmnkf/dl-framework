import sagemaker
from sagemaker.estimator import Estimator
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())

estimator = Estimator(image_uri='ipole/projection:latest',
                      role=os.environ['ROLE_ARN'],
                      instance_count=1,
                      instance_type='local'
                      )

estimator.fit()