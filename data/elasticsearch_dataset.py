import datatable as dt
from elasticsearch import Elasticsearch

from h2oaicore.data import CustomData

_global_modules_needed_by_name = ["elasticsearch"]

"""
Refs:
   https://elasticsearch-py.readthedocs.io/en/latest/api.html
   https://gist.github.com/hmldd/44d12d3a61a8d8077a3091c4ff7b9307

"""


class ElasticsearchData(CustomData):
    CLOUD_ID = "test-deployment:XXX-KEY"
    BASIC_AUTH = ("elastic", "XXX-PASS")
    INDEX = ".ds-logs-enterprise_search*"
    QUERY = {
        "query": {
            "match_all": {}
        }
    }
    COL_NAMES = ["id", "event", "timestamp"]

    @staticmethod
    def es_frame_generator(elastic_client, index, query, col_names):
        data = elastic_client.search(index=index, body=query, scroll="10m", size=10)
        sid = data['_scroll_id']
        scroll_size = len(data['hits']['hits'])
        while scroll_size > 0:
            for hit in data['hits']['hits']:
                yield dt.Frame(
                    [
                        [hit["_id"]],
                        [hit["_source"]["event"]["dataset"]],
                        [hit["_source"]["@timestamp"]]],
                    names=col_names,
                )
            data = elastic_client.scroll(scroll_id=sid, scroll='10m')
            sid = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

    @staticmethod
    def create_data(X: dt.Frame = None):
        elastic_client = Elasticsearch(
            cloud_id=ElasticsearchData.CLOUD_ID,
            basic_auth=ElasticsearchData.BASIC_AUTH,
        )

        dt_frame = dt.Frame(names=ElasticsearchData.COL_NAMES)

        for frame in ElasticsearchData.es_frame_generator(
                elastic_client,
                ElasticsearchData.INDEX,
                ElasticsearchData.QUERY,
                ElasticsearchData.COL_NAMES,
        ):
            dt_frame.rbind(frame)

        return {"ES_Sample": dt_frame}
