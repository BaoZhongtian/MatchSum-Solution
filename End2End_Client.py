import os
from json import load, dump
from rouge_score import rouge_scorer
from time import strftime, localtime, sleep
from paramiko import SSHClient, AutoAddPolicy, Transport, SFTPClient


class MatchSumClient:
    def __init__(self, ip, username, password, port, request_path=None,
                 result_path=None):
        assert request_path is not None and result_path is not None
        self.REQUEST_PATH = request_path
        self.RESULT_PATH = result_path

        self.UPLOAD_PATH = 'Request/'
        self.DOWNLOAD_PATH = 'Result/'
        if not os.path.exists(self.UPLOAD_PATH): os.makedirs(self.UPLOAD_PATH)

        print('Connecting To Server...')
        try:
            self.ssh = SSHClient()
            self.ssh.set_missing_host_key_policy(AutoAddPolicy())
            self.ssh.connect(ip, port, username, password)

            self.ftp_conn = Transport((ip, port))
            self.ftp_conn.connect(username=username, password=password)
            self.ftp_client = SFTPClient.from_transport(self.ftp_conn)
        except:
            print('!!!!!Connect Error!!!!!')
            os.system('pause')
            exit()
        print('Connect Completed.')

    def __search_file__(self, filename):
        while True:
            sleep(1)
            stdin, stdout, stderr = self.ssh.exec_command('dir ' + self.RESULT_PATH)
            stout = stdout.read().decode()
            if stout.find(filename) != -1: break

    @staticmethod
    def package_request(file_name, summary_name=None):
        time_ticket = strftime('%Y_%m_%d_%H_%M_%S', localtime())

        with open(file_name, 'r') as file:
            data = file.readlines()
        packaged_request = {'Time': time_ticket, 'Text': [_.replace('\n', '').lower() for _ in data]}
        if summary_name is not None:
            with open(summary_name, 'r') as file:
                data = file.readlines()
            packaged_request['Summary'] = [_.replace('\n', '').lower() for _ in data]
        return packaged_request

    def upload_file(self):
        if not os.path.exists(self.UPLOAD_PATH): os.makedirs(self.UPLOAD_PATH)
        if not os.path.exists(self.DOWNLOAD_PATH): os.makedirs(self.DOWNLOAD_PATH)

        if len(os.listdir(self.UPLOAD_PATH)) == 0: return
        for filename in os.listdir(self.UPLOAD_PATH):
            if filename[-4:] == 'json': break

        try:
            self.ftp_client.put(self.UPLOAD_PATH + filename, self.REQUEST_PATH + filename)
        except:
            print('!!!!!Could Not Upload Request!!!!!')
            os.system('pause')
            exit()
        print(filename + ' Upload Completed.')

        self.__search_file__(filename)
        print(filename + ' Summarization Treat Completed.')
        self.ftp_client.get(self.RESULT_PATH + filename, self.DOWNLOAD_PATH + filename)
        print(filename + ' Download Completed.')

        request = load(open(self.UPLOAD_PATH + filename, 'r'))
        result = load(open(self.DOWNLOAD_PATH + filename, 'r'))

        if 'Summary' in request.keys():
            summary = ' '.join(request['Summary'])
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        counter = 1
        while 'Top-%d' % counter in result.keys():
            print('Top-%d' % counter)
            print(result['Top-%d' % counter])
            if 'Summary' in request.keys():
                print('Rouge - 1 Score =',
                      scorer.score(target=summary, prediction=result['Top-%d' % counter])['rouge1'].fmeasure)
            print()
            counter += 1

        os.remove(self.UPLOAD_PATH + filename)


if __name__ == '__main__':
    client = MatchSumClient(ip='123.56.6.206', username='cide', password='1111', port=5053,
                            request_path='/home/cide/MatchSum/Request/', result_path='/home/cide/MatchSum/Result/')

    packaged = MatchSumClient.package_request('Demo/input.txt', 'Demo/input.summary')
    dump(packaged, open('Request/' + packaged['Time'] + '.json', 'w'))
    client.upload_file()
