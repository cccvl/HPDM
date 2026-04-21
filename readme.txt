This is the code repo for paper: Attack Face Forgery Video Detection System via Hybrid Perturbation Guided by Dual Mask.
In this paper, we attempt to attack deep forgery detection
systems rigorously to uncover their shortcomings. Unlike
existing methods that solely attack fake regions to increase their
authenticity, the proposed identifies decisive regions to decision
and creates a dual mask to carry out different attacks. The
dual mask comprises two components: a positive mask, which
identifies real regions for targeted attacks, and a negative mask,
pinpointing fake regions for disruption.Perturbations applied to
the positive mask aim to bolster confidence in real decisions,
while those on the negative mask aim to diminish confidence
in fake decisions, thereby inducing the detection system to
make erroneous judgments. Additionally, to enhance the visual
quality of the manipulated samples, we introduce a hybrid
constraint optimization, which carefully balances the advantages
and disadvantages of spatial perturbations and frequency-domain
adversarial samples, facilitating the learning of perturbations
that maintain high attack efficacy while preserving visual fidelity.
These two modules are integrated into a novel GAN-based
attacking network, HPDM, to generate adversarial samples and
achieve improvements in attack transferability and imperceptibility.
We evaluated our method through experiments on FFHQ and
FF++. The results demonstrate that our approach consistently
outperforms existing methods in terms of average attack efficacy.
