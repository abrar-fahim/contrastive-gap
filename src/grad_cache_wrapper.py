

from grad_cache.functional import cached, cat_input_tensor

import torch
import torch.nn.functional as F

from clips.clip_parent import ClipParent

class GradCacheWrapper():
    def __init__(self, clip_model: ClipParent) -> None:
        self.clip_model = clip_model

    
    @cached
    def get_text_projections(self, captions):
        # return representation after linear projection
        return self.clip_model.encode_text(captions)
    
    @cached
    def get_image_projections(self, preprocessed_images):
        # return representation after linear projection
        return self.clip_model.encode_image(preprocessed_images)
    

    @cat_input_tensor
    def contrastive_loss(self, image_projections, text_projections):
        '''
        image_projections: shape: (batch_size, 512)
        text_projections: shape: (batch_size, 512)
        need to incorporate temperature here
        keeping logit_scale fixed, and not a learnable parameter, for now
        '''
        # normalize
        # image_projections = F.normalize(image_projections, dim=1)
        # text_projections = F.normalize(text_projections, dim=1)

        # logit_scale = torch.log(torch.tensor([1 / temperature], device=image_projections.device)).item()

        image_projections = image_projections / torch.norm(image_projections, dim=1, keepdim=True)
        text_projections = text_projections / torch.norm(text_projections, dim=1, keepdim=True)

        logit_scale = self.clip_model.model.logit_scale.exp()

        print('logit_scale', logit_scale)

        # cosine similarity
        # shape: (batch_size, batch_size)
        similarity_matrix = logit_scale * torch.matmul(image_projections, text_projections.T)

        labels = torch.arange(similarity_matrix.shape[0]).to(similarity_matrix.device)

        # image_loss
        image_loss = F.cross_entropy(similarity_matrix, labels)

        # text_loss
        text_loss = F.cross_entropy(similarity_matrix.T, labels)

        loss = (image_loss + text_loss) / 2

        # # shape: (batch_size, )
        # image_similarity = torch.diag(similarity_matrix)

        # # shape: (batch_size, )
        # text_similarity = torch.diag(similarity_matrix.T)

        # # shape: (2 * batch_size, )
        # contrastive_similarity = torch.cat([image_similarity, text_similarity])

        # # shape: (2 * batch_size, )
        # labels = torch.arange(contrastive_similarity.shape[0] // 2)
        # labels = torch.cat([labels, labels])

        # # shape: (2 * batch_size, )
        # loss = F.cross_entropy(contrastive_similarity, labels)

        return loss