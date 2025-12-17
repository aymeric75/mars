from a7_datamanager.a7_processing.interface.client import A7Client, RDIClient

a7client = A7Client(
    "eyJraWQiOiIwMmE5MDllZGQyMjI3ZDhkMmI3NzczNjA2MGZkNjM5MjA1ZmJlMjc4IiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYifQ.eyJhdWQiOiJhNy1hcGkiLCJzdWIiOiJkZDhjNjI4NC1mYjI5LTQzMGEtYjZiNS1mNjI0Mjk0ZjQ3ZTIiLCJ0b2tlbl91c2UiOiJhcGkiLCJhdXRoX3RpbWUiOjE3NjE3NDUxNTgsImlzcyI6Imh0dHBzOi8vYTctdG9rZW4tc2VydmljZS5kZXV0c2NoZS1ib2Vyc2UuY29tIiwiY29tcGFueSI6IkRCQUciLCJleHAiOjE3Njk0NzIwMDAsImlhdCI6MTc2MTc0NTE1OCwiZW1haWwiOiJheW1lcmljLmJhcmJpbkB0dW5pLmZpIiwidXNlcm5hbWUiOiJheW1lYmFyIn0.W2q1WUSSTjhFHJ9AgpVThMlIcRUvLcCowp0xcy3CBET1zRCMPHu5fZJLnfui95wgnM-FG15SL_gOmBiwAVszC1dAetah2OosuOJW7XJ48fhUmpk0jL1NEuCWLdDehvEq_cDoVYIpVVYXs31RU8igGQHp3tJNZYmuo8Jm3y8ZRYxGJxi2f2L0ew2_UhxLVJanpYxM53oj76Tw1RNA6m2mBld0ljXj5fqPnsbnidvjTJjsmlVCO9YoMw13sO6Oi_ji4NSQZf1qMoIPXzg8zrh8FOGQHFR5CapO2FaPlPCTKa1E1gMAhhpFB-Y3RqWoW3c2wCeG0-v44uTdkwLVPkPU7A"
)

rdiclient = RDIClient(a7client)


# 1033420/12712237/20250901

print(rdiclient.get_security_details("XETR", "20250901", "1033420", "12712237"))

# import torch as th

# from market_simulation.models.order_model import OrderModel

# model = OrderModel(
#     emb_dim=1024,
#     num_layers=24,
#     num_heads=16,
#     num_bins_price_level=32,
#     num_bins_pred_order_volume=32,
#     num_bins_order_interval=16,
#     num_max_orders=1024,  # sequence length
# ).eval()


# B = 2
# seq_len = 1024
# token_dim = 15

# # IMPORTANT: indices should be integer types (embeddings expect ints)
# x = th.zeros((B, seq_len, token_dim), dtype=th.long)
