from spektral.data import Dataset, Graph, DisjointLoader
from src.graph_network_v5 import GraphNetworkV5


class CounterDataset(Dataset):
    def __init__(self, **kwargs):
        self.og_data = kwargs['og_data']
        self.y_label = kwargs['y_label']
        super().__init__(**kwargs)
        
    def read(self):
        data = self.og_data
        data_mat = data['normal']

        return [
            Graph(x=x, a=a, e=e, y=y) for x, a, e, y in zip(
                data_mat['x'], data_mat['a'], data_mat['e'], data['binary']
            )
        ]

        
class CounterAttackModel:
    '''
    Obtain the raw data, convert into CounterDataset - a variation of Dataset class of spektral library.
    '''
    def __init__(self, model, kwargs):
        include_ball_node = True
        self_loop_ball = True  # connect ball node to its self
        connect_via = 'ball'  # connect via 'ball_carrier' node or via 'ball' node
        include_acceleration_features = False
        include_intended_receiver = True
        include_intended_receiver_node_feature = True
        include_historic_features = False
        self._model = model
        
        self._graph_network = GraphNetworkV5(
            kwargs = kwargs,
            include_ball_node=include_ball_node,
            self_loop_ball=self_loop_ball,
            connect_via=connect_via,
            include_acceleration_features=include_acceleration_features,
            include_intended_receiver=include_intended_receiver,
            include_intended_receiver_node_feature=include_intended_receiver_node_feature,
            include_historic_features=include_historic_features
        )
        
    def get_counter_probability(self):
        '''
        Return the predicted probability for the given graph.
        '''
        self._graph_network.compute_features()
        # Get raw graph data
        dataset_1 = self._graph_network.get_graph_data()
        
        data_1 = CounterDataset(og_data = dataset_1, y_label='binary')
        loader_te = DisjointLoader(data_1, batch_size=1, epochs = 1, shuffle = False)

        for batch in loader_te:
            inputs, target = batch
            p = self._model(inputs, training=False)
            return p.numpy()[0][0]
        
        return None