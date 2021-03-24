import pandas as pd
import pickle
from ibm_botocore.client import Config
#from boto3.s3.transfer import S3Transfer, TransferConfig
import ibm_boto3
import os
class COS_File_Loader(object):
    def __init__(self, cos_credentials):
        self.cos_credentials = cos_credentials
        self.cos_client = ibm_boto3.client(service_name='s3',
            ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
            ibm_service_instance_id=cos_credentials['IAM_SERVICE_ID'],
            ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
            config=Config(signature_version='oauth'),
            endpoint_url=cos_credentials['ENDPOINT'])
        self.cos_resource = ibm_boto3.resource(service_name='s3',
            ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
            ibm_service_instance_id=cos_credentials['IAM_SERVICE_ID'],
            ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
            config=Config(signature_version='oauth'),
            endpoint_url=cos_credentials['ENDPOINT'])
    def download_file(self,file_name):
        try:
            self.cos_client.download_file(Bucket=self.cos_credentials['BUCKET'],Key=file_name,Filename=file_name)
        except Exception as e:
            print(Exception, e)
    def upload_file(self,file_name):
        try:
            self.cos_client.upload_file(file_name, Bucket=self.cos_credentials['BUCKET'],Key=file_name)
        except Exception as e:
            print(Exception, e)
    def load_csv(self,file_name): 
        self.download_file(file_name)
        contents = pd.read_csv(file_name, sep=',')
        return contents
    def save_csv(self,df,file_name):     
        df.to_csv(file_name, sep=',',header=True, index=False)
        self.upload_file(file_name)
        return 
    def load_pkl(self,file_name):  
        self.download_file(file_name)
        contents = pickle.load(open(file_name, 'rb'))
        return contents
    def save_pkl(self,file_name, save_object):
        pickle.dump(save_object,open(file_name,'wb'))
        self.upload_file(file_name)
    def test_save_load_pkl(self):    
        print ()
        print ('Testing PKL file')
        save_object = [1,2,3]
        file_name = "test_save_object.pkl"
        print ('Before Save')
        print (save_object)
        self.save_pkl(file_name, save_object)  
        os.remove("test_save_object.pkl")
        new_object = self.load_pkl(file_name)  
        print ('After Load')
        print (new_object)
        return new_object
    def test_save_load_csv(self):   
        print
        print ('Testing CSV file')
        my_csv = [[1,2],[3,4]]
        df_test = pd.DataFrame(my_csv,columns=['a','b'])
        print ('Before Save')
        print (df_test.head())
        self.save_csv(df_test,'test.csv')
        os.remove('test.csv')
        df_test = None
        df_test = self.load_csv('test.csv')
        print ('After Load')
        print (df_test.head())
        return df_test
    def get_bucket_contents(self):
        bucket_name = self.cos_credentials['BUCKET']
        print("Retrieving bucket contents from: {0}".format(bucket_name))
        try:
            files = self.cos_resource.Bucket(bucket_name).objects.all()
            for file in files:
                print("Item: {0} ({1} bytes).".format(file.key, file.size))
        except Exception as e:
            print("Unable to retrieve bucket contents: {0}".format(e))
        return 
