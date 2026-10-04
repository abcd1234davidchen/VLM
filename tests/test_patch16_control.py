"""Control auxiliary isolation and aligned comparison contracts."""
import unittest
import torch

from vlm.training.patch16_skeleton import auxiliary
from vlm.training.patch16_control_compare import align_predictions, paired_counts


class ControlContracts(unittest.TestCase):
    def test_auxiliary_is_disconnected_even_with_invalid_skeleton_values(self):
        maps=torch.rand(2,8,2,198,requires_grad=True)
        extra=auxiliary(maps,torch.tensor([0,1]),torch.full((2,8,196),float('nan')),
                        torch.ones(2,8,dtype=torch.bool),'ce_retention',None)
        self.assertEqual(float(extra),0)
        self.assertFalse(extra.requires_grad)
        scores=torch.randn(2,120,requires_grad=True)
        (torch.nn.functional.cross_entropy(scores,torch.tensor([0,1]))+extra).backward()
        self.assertIsNone(maps.grad)
        self.assertGreater(float(scores.grad.abs().sum()),0)

    def test_aligned_permutation_and_paired_counts(self):
        ids=['a','b','c','d'];labels=torch.tensor([0,1,2,3]);logits=torch.randn(4,120)
        permutation=torch.tensor([2,0,3,1])
        payload={'sample_ids':[ids[i] for i in permutation], 'labels':labels[permutation],
                 'logits':{'whole':logits[permutation],'center_zoom70':logits[permutation]},'test_evaluated':False}
        aligned=align_predictions(payload,ids,labels)
        self.assertTrue(torch.equal(aligned['whole'],logits))
        counts,_=paired_counts(torch.tensor([0,4,2,9]),torch.tensor([4,1,2,9]),labels)
        self.assertEqual(counts,{'native_only_correct':1,'control_only_correct':1,'both_correct':1,'both_wrong':1})
        payload['labels'][0]=99
        with self.assertRaises(ValueError):align_predictions(payload,ids,labels)

    def test_duplicate_ids_and_nonfinite_predictions_fail_closed(self):
        logits=torch.zeros(2,120);ids=['a','b'];labels=torch.tensor([0,1])
        payload={'sample_ids':['a','a'],'labels':labels,'logits':{'whole':logits,'center_zoom70':logits},'test_evaluated':False}
        with self.assertRaises(ValueError):align_predictions(payload,ids,labels)
        payload['sample_ids']=ids;payload['logits']['whole'][0,0]=float('nan')
        with self.assertRaises(ValueError):align_predictions(payload,ids,labels)


if __name__=='__main__':unittest.main()
