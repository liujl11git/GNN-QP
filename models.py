import tensorflow as tf
import tensorflow.keras as K
import pickle


class BipartiteGraphConvolution(K.Model):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).

    Args:
        emb_size (int): Hidden embedding size of the model
        activation (tf.keras.activations): Activation function
        initializer (tf.keras.initializers): Initializer for dense layers
        right_to_left (bool): If ``True``, gather vertex features and scatter
            to constraint nodes. Otherwise, gather constraint features and
            scatter to vertex nodes.
        gather_first (bool): If ``True``, the resulting model corresponds to
            the mixed-integer extension in the paper.
    """

    def __init__(self, emb_size, activation, initializer,
                 right_to_left=False, gather_first=False):
        super().__init__()

        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        self.gather_first = gather_first

        # feature layers
        self.feature_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])
        # output_layers
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])

    def build(self, input_shapes):

        l_shape, ei_shape, ev_shape, r_shape = input_shapes

        fshape = r_shape if self.right_to_left else l_shape
        if self.gather_first:
            fshape = (None, fshape[1] + ev_shape[1])
        self.feature_module.build(fshape)

        self.output_module.build([None, self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1])])
        self.built = True

    def call(self, inputs):

        left_features, edge_indices, edge_features, right_features = inputs

        if self.right_to_left:
            scatter_dim, gather_dim = 0, 1
            prev_features, gather_feautures = left_features, right_features
        else:
            scatter_dim, gather_dim = 1, 0
            prev_features, gather_feautures = right_features, left_features

        # compute joint features
        if self.gather_first:
            gathered = tf.gather(gather_feautures, axis=0, indices=edge_indices[gather_dim])
            joint_features = self.feature_module(tf.concat([edge_features, gathered], axis=1))
        else:
            gather_feautures = self.feature_module(gather_feautures)
            joint_features = edge_features * tf.gather(
                gather_feautures, axis=0, indices=edge_indices[gather_dim]
            )

        # perform convolution
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[prev_features.shape[0], self.emb_size]
        )

        # output layer
        output = self.output_module(tf.concat([conv_output, prev_features,], axis=1))

        return output


class ExtendedBipartiteGraphConvolution(K.Model):
    """
    Extended partial bipartite graph convolution (either left-to-right or
    right-to-left), where we allow connections within one side of the bipartition.

    Args:
        emb_size (int): Hidden embedding size of the model
        activation (tf.keras.activations): Activation function
        initializer (tf.keras.initializers): Initializer for dense layers
        right_to_left (bool): If ``True``, gather vertex features and scatter
            to constraint nodes. Otherwise, gather constraint features and
            scatter to vertex nodes.
        gather_first (bool): If ``True``, the resulting model corresponds to
            the mixed-integer extension in the paper.
    """

    def __init__(self, emb_size, activation, initializer, right_to_left=False, gather_first=False):
        super().__init__()

        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        self.gather_first = gather_first

        # feature layers
        self.feature_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])
        self.feature_module_self = K.Sequential([
            K.layers.Dense(units=self.emb_size, use_bias=False, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])

        # output_layers
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])

    def build(self, input_shapes):

        l_shape, ei_shape, ev_shape, r_shape = input_shapes

        if self.right_to_left:
            if self.gather_first:
                self.feature_module.build((None, r_shape[1] + ev_shape[1]))
                self.feature_module_self.build((None, l_shape[1] + ev_shape[1]))
            else:
                self.feature_module.build(r_shape)
                self.feature_module_self.build(l_shape)
        else:
            if self.gather_first:
                self.feature_module.build((None, l_shape[1] + ev_shape[1]))
                self.feature_module_self.build((None, r_shape[1] + ev_shape[1]))
            else:
                self.feature_module.build(l_shape)
                self.feature_module_self.build(r_shape)
        self.output_module.build(
            [None, self.emb_size*2 + (l_shape[1] if self.right_to_left else r_shape[1])]
        )
        self.built = True

    def call(self, inputs):

        left_features, edge_indices, edge_features, right_features, self_edges = inputs

        if self.right_to_left:
            scatter_dim, gather_dim = 0, 1
            prev_features, gather_feautures = left_features, right_features
        else:
            scatter_dim, gather_dim = 1, 0
            prev_features, gather_feautures = right_features, left_features

        # compute joint features
        if self.gather_first:
            gathered = tf.gather(gather_feautures, axis=0, indices=edge_indices[gather_dim])
            joint_features = self.feature_module(tf.concat([edge_features, gathered], axis=1))
        else:
            gather_features = self.feature_module(gather_feautures)
            joint_features = edge_features * tf.gather(
                gather_features, axis=0, indices=edge_indices[gather_dim]
            )

        # perform convolution
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[prev_features.shape[0], self.emb_size]
        )

        # self convolution
        if isinstance(self_edges, tf.Tensor):
            if self.gather_first:
                raise ValueError("Q in matrix is not supported for mixed-integer extension.")
            self_out = self.feature_module_self(prev_features)
            self_out = tf.linalg.matmul(self_edges, self_out)
        elif isinstance(self_edges, tuple) and len(self_edges) == 2:
            self_inds, self_feats = self_edges
            if self.gather_first:
                self_gathered = tf.gather(prev_features, axis=0, indices=self_inds[1])
                self_gathered = self.feature_module_self(tf.concat([self_feats, self_gathered], axis=1))
            else:
                self_gathered = self_feats * tf.gather(
                    self.feature_module_self(prev_features),
                    axis=0,
                    indices=self_inds[1]
                )
            self_out = tf.scatter_nd(
                updates=self_gathered,
                indices=tf.expand_dims(self_inds[0], axis=1),
                shape=[prev_features.shape[0], self_gathered.shape[1]]
            )

        # output layer
        to_concat = [prev_features, conv_output,]
        if self_edges is not None:
            to_concat += [self_out]
        output = self.output_module(tf.concat(to_concat, axis=1))

        return output


class GCNPolicy(K.Model):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.

    Args:
        embSize (int): dimension of the hidden embeddings
        nConsF (int): dimension of constraint features
        nEdgeF (int): dimension of edge features
        nVarF (int): dimension of variable features
        isGraphLevel (bool): takes `True` if each graph has a scalar output value
    """

    def __init__(self, embSize, nLayers, nConsF, nEdgeF, nVarF,
                 gather_first=False, isGraphLevel=True):
        super().__init__()

        self.emb_size = embSize
        self.num_layers = nLayers
        self.cons_nfeats = nConsF
        self.edge_nfeats = nEdgeF
        self.var_nfeats = nVarF
        self.gather_first = gather_first
        self.is_graph_level = isGraphLevel

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()
        # self.initializer = K.initializers.RandomNormal(mean=0.0, stddev=0.05)

        # CONSTRAINT EMBEDDING
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size,
                           activation=self.activation,
                           kernel_initializer=self.initializer),
        ])

        # EDGE EMBEDDING
        self.edge_embedding = K.Sequential([])

        # VARIABLE EMBEDDING
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size,
                           activation=self.activation,
                           kernel_initializer=self.initializer),
        ])

        # GRAPH CONVOLUTIONS
        self.v_to_c_layers = []
        self.c_to_v_layers = []
        for i in range(self.num_layers):
            self.v_to_c_layers.append(
                BipartiteGraphConvolution(
                    self.emb_size, self.activation, self.initializer,
                    right_to_left=True, gather_first=self.gather_first,
                )
            )
            self.c_to_v_layers.append(
                ExtendedBipartiteGraphConvolution(
                    self.emb_size, self.activation, self.initializer,
                    gather_first=self.gather_first,
                )
            )

        # OUTPUT
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, kernel_initializer=self.initializer, use_bias=False),
        ])

        # build model right-away
        self.build([
            (None, self.cons_nfeats),
            (2, None),
            (None, self.edge_nfeats),
            (None, self.var_nfeats),
            (None, ),
            (None, ),
        ])

        # save / restore fix
        self.variables_topological_order = [v.name for v in self.variables]

        # save input signature for compilation
        self.input_signature = [
            (
                tf.TensorSpec(shape=[None, self.cons_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[2, None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, self.edge_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[None, self.var_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
            ),
            tf.TensorSpec(shape=[], dtype=tf.bool),
        ]

    def build(self, input_shapes):

        c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.cons_embedding.build(c_shape)
            self.var_embedding.build(v_shape)
            for v_to_c in self.v_to_c_layers:
                v_to_c.build((emb_shape, ei_shape, ev_shape, emb_shape))
            for c_to_v in self.c_to_v_layers:
                c_to_v.build((emb_shape, ei_shape, ev_shape, emb_shape))
            if self.is_graph_level:
                self.output_module.build([None, 2 * self.emb_size])
            else:
                self.output_module.build([None, 3 * self.emb_size])
            self.built = True

    def call(self, inputs, training, cons_segments=None, var_segments=None, var_repeats=None):

        (constraint_features, edge_indices, edge_features, variable_features,
         Q_matrix,) = inputs

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)

        # Graph convolutions
        for v_to_c, c_to_v in zip(self.v_to_c_layers, self.c_to_v_layers):
            constraint_features = v_to_c((constraint_features, edge_indices, edge_features, variable_features))
            constraint_features = self.activation(constraint_features)
            variable_features = c_to_v((constraint_features, edge_indices, edge_features, variable_features, Q_matrix))
            variable_features = self.activation(variable_features)

        if cons_segments is not None and var_segments is not None:
            constraint_features_mean = tf.math.segment_mean(constraint_features, cons_segments)
            variable_features_mean = tf.math.segment_mean(variable_features, var_segments)
            final_features = tf.concat([variable_features_mean, constraint_features_mean], axis=1)
        else:
            var_repeats = [variable_features.shape[0]]
            variable_features_mean = tf.reduce_mean(variable_features, axis=0, keepdims=True)
            constraint_features_mean = tf.reduce_mean(constraint_features, axis=0, keepdims=True)
            final_features = tf.concat([variable_features_mean, constraint_features_mean], axis=1)

        if not self.is_graph_level:
            global_features = tf.repeat(final_features, var_repeats, axis=0)
            final_features = tf.concat([variable_features, global_features], axis=1)

        # OUTPUT
        output = self.output_module(final_features)

        return output

    def save_state(self, path):
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    def restore_state(self, path):
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))
