import torch
import torch.nn as nn
import positional_encoder as pe


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, 
        dec_seq_len, 
        batch_first, 
        out_seq_len=58, 
        dim_val=512, 
        n_encoder_layers=4, 
        n_decoder_layers=4, 
        n_heads=8, 
        dropout_encoder=0.2, 
        dropout_decoder=0.2, 
        dropout_pos_enc=0.1, 
        dim_feedforward_encoder=2048, 
        dim_feedforward_decoder=2048, 
        num_predicted_features=1):


        super().__init__()
        self.dec_seq_len = dec_seq_len

        #create linear layers needed for model
        self.encoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=dim_val)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=num_predicted_features)

        #create positional encoder
        self.positional_encoding_layer = pe.PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder)

        #stack the encoder layer n times
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_decoder, dropout=dropout_decoder)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        #pass through input layer right before encoder (embedding layer)
        src = self.encoder_input_layer(src)

        #pass through positional encoder layer
        src = self.positional_encoding_layer(src)

        #pass through all stacked encoder layer in encoder
        src = self.encoder(src=src)

        #pass decoder input through decoder input embedding layer
        decoder_embedding_output = self.decoder_input_layer(tgt)

        #pass through decoder
        decoder_output = self.decoder(tgt=decoder_embedding_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)

        #pass through final linear mapping
        final_output = self.linear_mapping(decoder_output)

        return final_output
