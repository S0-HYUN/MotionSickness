import torch
import torch.nn as nn

# class QuadrupletLossModel(nn.Module):
    
#     def __init__(self, resnet: nn.Module):
#         super().__init__()
#         self.resnet = resnet
#         self.resnet.fc = nn.Identity()
#         self.embeddings = nn.Linear(512, 128)
        
#     def forward(
#             self, 
#             inputs: torch.Tensor,  # [B, C, H, W]
#             labels: torch.Tensor  # [B]
#         ):
#         B = labels.size(0)
#         embeddings = self.embeddings(self.resnet(inputs))  # [B, E]
#         distance_matrix = get_distance_matrix(embeddings)  # [B, B]
#         with torch.no_grad():
#             mask_pos = get_positive_mask(labels, device)  # [B, B]
#             mask_neg = get_negative_mask(labels, device)  # [B, B]
#             triplet_mask = get_triplet_mask(labels, device)  # [B, B, B]
#             quadruplet_mask = get_quadruplet_mask(labels, device)  # [B, B, B, B]
#             unmasked_triplets = torch.sum(triplet_mask)  # [1]
#             unmasked_quadruplets = torch.sum(quadruplet_mask)  # [1]
#             mu_pos = torch.mean(distance_matrix[mask_pos])  # [1]
#             mu_neg = torch.mean(distance_matrix[mask_neg])  # [1]
#             mu = mu_neg - mu_pos  # [1]
        
#         distance_i_j = distance_matrix.view(B, B, 1)  # [B, B, 1]
#         distance_i_k = distance_matrix.view(B, 1, B)  # [B, 1, B]
#         triplet_loss_unmasked = distance_i_k - distance_i_j   # [B, B, B]
#         triplet_loss_unmasked = triplet_loss_unmasked[triplet_mask] # [valid_triplets]
#         hardest_triplets = triplet_loss_unmasked < max(mu, 0)  # [valid_triplets]
#         triplet_loss = triplet_loss_unmasked[hardest_triplets]  # [valid_triplets_after_mask]
#         triplet_loss = nn.functional.relu(triplet_loss)  # [valid_triplets_after_mask]

#         distance_i_j = distance_matrix.view(B, B, 1, 1)  # [B, B, 1, 1]
#         distance_k_l = distance_matrix.view(1, 1, B, B)  # [1, 1, B, B]
#         auxilary_loss_unmasked = distance_k_l - distance_i_j  # [B, B, B, B]
#         auxilary_loss_unmasked = auxilary_loss_unmasked[quadruplet_mask]  # [valid_quadruplets]
#         hardest_quadruples = auxilary_loss_unmasked < max(mu, 0)/2  # [valid_quadruplets_after_mask]
#         auxilary_loss = auxilary_loss_unmasked[hardest_quadruples]  # [valid_quadruplets_after_mask]
#         auxilary_loss = nn.functional.relu(auxilary_loss)  # [valid_triplets_after_mask]

#         quadruplet_loss = triplet_loss.mean() + auxilary_loss.mean()
#         logs = {
#             'positive_pairs': torch.sum(mask_pos).cpu().detach().item(),
#             'negative_pairs': torch.sum(mask_neg).cpu().detach().item(),
#             'mu_neg': mu_neg.cpu().detach().item(),
#             'mu_pos': mu_pos.cpu().detach().item(),
#             'valid_triplets': unmasked_triplets.cpu().detach().item(),
#             'valid_triplets_after_mask': triplet_loss.size(0),
#             'valid_quadruplets': unmasked_quadruplets.cpu().detach().item(),
#             'valid_quadruplets_after_mask': auxilary_loss.size(0),
#             'auxilary_loss': auxilary_loss.mean().cpu().detach().item(),
#             'triplet_loss': triplet_loss.mean().cpu().detach().item()
#         }
#         return quadruplet_loss, logs

class QuadrupletLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embeddings, labels, device):
        labels = torch.tensor(labels).to(device)
        B = labels.size(0)
        distance_matrix = get_distance_matrix(embeddings)
        mask_pos = get_positive_mask(labels, device)  # [B, B]
        mask_neg = get_negative_mask(labels, device)  # [B, B]
        triplet_mask = get_triplet_mask(labels, device)  # [B, B, B]
        quadruplet_mask = get_quadruplet_mask(labels, device)  # [B, B, B, B]
        unmasked_triplets = torch.sum(triplet_mask)  # [1]
        unmasked_quadruplets = torch.sum(quadruplet_mask)  # [1]
        mu_pos = torch.mean(distance_matrix[mask_pos])  # [1]
        mu_neg = torch.mean(distance_matrix[mask_neg])  # [1]
        mu = mu_neg - mu_pos  # [1]

        distance_i_j = distance_matrix.view(B, B, 1)  # [B, B, 1]
        distance_i_k = distance_matrix.view(B, 1, B)  # [B, 1, B]
        triplet_loss_unmasked = distance_i_k - distance_i_j   # [B, B, B]
        triplet_loss_unmasked = triplet_loss_unmasked[triplet_mask] # [valid_triplets]
        hardest_triplets = triplet_loss_unmasked < max(mu, 0)  # [valid_triplets]
        triplet_loss = triplet_loss_unmasked[hardest_triplets]  # [valid_triplets_after_mask]
        triplet_loss = nn.functional.relu(triplet_loss)  # [valid_triplets_after_mask]

        distance_i_j = distance_matrix.view(B, B, 1, 1)  # [B, B, 1, 1]
        distance_k_l = distance_matrix.view(1, 1, B, B)  # [1, 1, B, B]
        auxilary_loss_unmasked = distance_k_l - distance_i_j  # [B, B, B, B]
        auxilary_loss_unmasked = auxilary_loss_unmasked[quadruplet_mask]  # [valid_quadruplets]
        hardest_quadruples = auxilary_loss_unmasked < max(mu, 0)/2  # [valid_quadruplets_after_mask]
        auxilary_loss = auxilary_loss_unmasked[hardest_quadruples]  # [valid_quadruplets_after_mask]
        auxilary_loss = nn.functional.relu(auxilary_loss)  # [valid_triplets_after_mask]

        quadruplet_loss = triplet_loss.mean() + auxilary_loss.mean()
        return quadruplet_loss


def get_distance_matrix(
        embeddings: torch.Tensor,  #  [B, E]
    ):
    B = embeddings.size(0)
    dot_product = embeddings @ embeddings.T  # [B, B]
    squared_norm = torch.diag(dot_product) # [B]
    distances = squared_norm.view(1, B) - 2.0 * dot_product + squared_norm.view(B, 1)  # [B, B]
    return torch.sqrt(nn.functional.relu(distances) + 1e-16)  # [B, B]

def get_positive_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    return labels_equal & ~indices_equal  # [B, B]

def get_negative_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    return ~labels_equal & ~indices_equal  # [B, B]

def get_triplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):

    B = labels.size(0)

    # Make sure that i != j != k
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    indices_not_equal = ~indices_equal  # [B, B]
    i_not_equal_j = indices_not_equal.view(B, B, 1)  # [B, B, 1]
    i_not_equal_k = indices_not_equal.view(B, 1, B)  # [B, 1, B]
    j_not_equal_k = indices_not_equal.view(1, B, B)  # [1, B, B]
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k  # [B, B, B]

    # Make sure that labels[i] == labels[j] but labels[i] != labels[k]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1)  # [B, B, 1]
    i_equal_k = labels_equal.view(B, 1, B)  # [B, 1, B]
    valid_labels = i_equal_j & ~i_equal_k  # [B, B, B]

    return distinct_indices & valid_labels  # [B, B, B]

def get_quadruplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)

    # Make sure that i != j != k != l
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B] 
    indices_not_equal = ~indices_equal  # [B, B] 
    i_not_equal_j = indices_not_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_not_equal_k = indices_not_equal.view(1, B, B, 1)  # [B, 1, 1, B] 
    k_not_equal_l = indices_not_equal.view(1, 1, B, B)  # [1, 1, B, B] 
    distinct_indices = i_not_equal_j & j_not_equal_k & k_not_equal_l  # [B, B, B, B] 

    # Make sure that labels[i] == labels[j] 
    #            and labels[j] != labels[k] 
    #            and labels[k] != labels[l]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_equal_k = labels_equal.view(1, B, B, 1)  # [1, B, B, 1]
    k_equal_l = labels_equal.view(1, 1, B, B)  # [1, 1, B, B]
    
    return (i_equal_j & ~j_equal_k & ~k_equal_l) & distinct_indices  # [B, B, B, B] 