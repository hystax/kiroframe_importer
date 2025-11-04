import json
import logging
from abc import ABCMeta
import requests
from retrying import retry

LOG = logging.getLogger(__name__)


def retry_if_connection_error(exception):
    if isinstance(exception, requests.ConnectionError):
        return True
    if isinstance(exception, requests.HTTPError):
        if exception.response.status_code in (503,):
            return True
    return False


class AbstractHttpProvider(metaclass=ABCMeta):
    def __init__(self, token="", secret="", ip=None):
        self._token = token
        self._secret = secret
        self._ip = ip

    @property
    def headers(self):
        headers = {"Content-type": "application/json"}
        if self._token:
            headers.update({"x-api-key": str(self._token)})
        if self._ip:
            headers.update({"X-Forwarded-For": self._ip})
        return headers

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, value):
        self._token = value

    @property
    def ip(self):
        return self._ip


class RequestsHttpProvider(AbstractHttpProvider):
    def __init__(self, url, token="", secret="", verify=True, ip=None):
        self.url = url
        self.verify = verify
        self.session = requests.session()
        super().__init__(token, secret, ip)

    @retry(
        stop_max_delay=10000,
        wait_fixed=1000,
        retry_on_exception=retry_if_connection_error,
    )
    def request(self, path, method, data=None):
        full_url = self.url + path
        response = self.session.request(
            method, full_url, data=data, headers=self.headers, verify=self.verify
        )
        response.raise_for_status()
        response_body = None
        if response.status_code != requests.codes.no_content:
            response_body = json.loads(response.content.decode("utf-8"))
        return response.status_code, response_body

    def close(self):
        self.session.close()


class ArceeMiniCl:
    def __init__(
        self,
        address="127.0.0.1",
        port="80",
        api_version="v2",
        url=None,
        http_provider=None,
        token="",
        secret="",
        verify=True,
        ip=None,
    ):
        if http_provider is None:
            if url is None:
                url = "http://%s:%s" % (address, port)
            http_provider = RequestsHttpProvider(url, token, secret, verify, ip)
        self._http_provider = http_provider
        self._api_version = api_version

    def _url(self, sub_url):
        return "/arcee/%s/%s" % (self._api_version, sub_url)

    def _request(self, url, method, body=None):
        data = None
        if body is not None:
            data = json.dumps(body)
        return self._http_provider.request(self._url(url), method, data)

    def post(self, url, body):
        return self._request(url, "POST", body)

    @property
    def token(self):
        return self._http_provider.token

    @token.setter
    def token(self, value):
        self._http_provider.token = value

    @staticmethod
    def import_url():
        url = "import"
        return url

    def import_data(self, import_source, import_source_id, import_type, owner_id, line):
        b = {
            "import_source": import_source,
            "import_source_id": import_source_id,
            "import_type": import_type,
            "owner_id": owner_id,
            "line": line,
        }
        return self.post(self.import_url(), b)
