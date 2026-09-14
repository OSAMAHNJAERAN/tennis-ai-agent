import numpy as np
from scripts.evaluate.diagnose_wasb_proposal_response import response_stats, category


def test_temporal_attenuation_requires_target_response_and_uses_native_mapping():
    maps=np.zeros((3,12,12),np.float32)
    maps[0,5,5]=.5
    inverse=np.array([[2.,0,0],[0,2.,0]])
    stats=response_stats(maps,inverse,(20,30,44,54),(30,40),1024,576)
    assert stats['tolerance_pixel_count']>0
    assert stats['average_target_active_pixels']==0
    assert category([stats])=='TEMPORAL_ATTENUATION'
    maps[:,5,5]=.5
    assert category([response_stats(maps,inverse,(20,30,44,54),(30,40),1024,576)])=='COMPONENT_DISPLACEMENT'


def test_padding_and_distant_distractors_do_not_supply_target_evidence():
    maps=np.zeros((3,12,12),np.float32)
    maps[:,11,11]=.99
    inverse=np.array([[1.,0,0],[0,1.,0]])
    stats=response_stats(maps,inverse,(0,0,10,10),(2,2),512,288)
    assert category([stats])=='WEAK_RESPONSE'
    assert stats['average_global_peak']==0
    assert 'average_target_peak' not in response_stats(maps,inverse,(0,0,10,10),None,512,288)
