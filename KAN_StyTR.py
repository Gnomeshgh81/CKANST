import torch
import torch.nn as nn
from module.basic_model import BasicModel
from utils.utils import normal


class KanST(BasicModel):
    def __init__(self, vgg, encoder_c, encoder_s, Kan_decoder ,decoder, HSI, args):
        super().__init__()

        self.encoder_c = encoder_c
        self.encoder_s = encoder_s
        self.Kan_decoder = Kan_decoder
        self.decoder = decoder
        self.HSI = HSI

        self.args = args

        # trained VGG
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        if self.args.Train:
            self.optimizerG = torch.optim.Adam([
                              {'params': self.encoder_c.parameters()},
                              {'params': self.encoder_s.parameters()},
                              {'params': self.Kan_decoder.parameters()},
                              {'params': self.decoder.parameters()},
                              ], lr=args.lr)

    def adjust_learning_rate(self, optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = 2e-4 / (1.0 + self.args.lr_decay * (iteration_count - 1e4))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_learning_rate(self, optimizer, iteration_count):
        """Imitating the original implementation"""
        lr = self.args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
        # print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def adain(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)



    def norm_opt(self, samples_c, samples_s):
        if self.args.Train:
            self.set_requires_grad([self.encoder_c, self.encoder_s, self.Kan_decoder, self.decoder, self.HSI], True)
            self.set_requires_grad([self.encoder_c, self.encoder_s, self.Kan_decoder, self.decoder], True)

        content_input = samples_c
        style_input = samples_s

        f_c = self.encoder_c(content_input)
        f_s = self.encoder_s(style_input)
        Ics = self.decoder(self.Kan_decoder(self.HSI(f_c, f_s)))

        if self.args.Train:
            Icc = self.decoder(self.Kan_decoder(self.HSI(f_c, f_c)))
            Iss = self.decoder(self.Kan_decoder(self.HSI(f_s, f_s)))

            content_feats = self.encode_with_intermediate(samples_c)
            style_feats = self.encode_with_intermediate(samples_s)
            Ics_feats = self.encode_with_intermediate(Ics)
            Icc_feats = self.encode_with_intermediate(Icc)
            Iss_feats = self.encode_with_intermediate(Iss)

            loss_c = (self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) +
                      self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2])))

            loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
            for i in range(1, 5):
                loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

            # Identity losses lambda 1
            loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)

            # Identity losses lambda 2
            loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(
                Iss_feats[0], style_feats[0])
            for i in range(1, 5):
                loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(
                    Iss_feats[i], style_feats[i])

            loss = loss_c * self.args.content_weight + loss_s * self.args.style_weight + loss_lambda1 * 70 + loss_lambda2

            self.optimizerG.zero_grad()
            loss.sum().backward()
            self.optimizerG.step()
            return Ics, Icc, Iss, loss_c, loss_s, loss_lambda1, loss_lambda2
        else:
            return Ics

    def forward(self, samples_c, samples_s, i):
        if self.args.Train:
            if i < 1e4:
                self.warmup_learning_rate(self.optimizerG, iteration_count=i)
            else:
                self.adjust_learning_rate(self.optimizerG, iteration_count=i)
            Ics,Icc, Iss, loss_c, loss_s, loss_lambda1, loss_lambda2 = self.norm_opt(samples_c,samples_s)
            return Ics,Icc, Iss, loss_c, loss_s, loss_lambda1, loss_lambda2
        else:
            Ics = self.norm_opt(samples_c, samples_s)
            return Ics
