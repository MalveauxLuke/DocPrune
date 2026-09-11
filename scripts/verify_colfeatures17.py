"""Verify exported features without loading a model or GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from colfeatures_package import inventory


def verify(root, cases=17):
    completion = json.loads((root / 'completion.json').read_text())
    files = [r for r in inventory(root) if r['path'] != 'completion.json']
    if completion['files'] != files or completion['cases'] != cases or completion['pages'] != 4 * cases:
        raise ValueError('Completion count or artifact integrity mismatch')
    required = {'embeddings', 'input_ids', 'attention_mask', 'pre_projection_hidden'}
    for ordinal in range(1, cases + 1):
        folder = root / 'features' / f'Q{ordinal:02d}'
        with np.load(folder / 'query.npz', allow_pickle=False) as data:
            if not required <= set(data.files):
                raise ValueError('Incomplete query features')
            query = data['embeddings']
            if not np.isfinite(query).all():
                raise ValueError('Non-finite query features')
        for rank in range(4):
            with np.load(folder / f'page-{rank:02d}.npz', allow_pickle=False) as data:
                if not required | {'pixel_values', 'image_grid_thw', 'image_positions',
                        'merged_vision_features', 'region_patch_overlap', 'region_query_maxsim',
                        'query_token_by_page_token', 'patch_boxes_normalized'} <= set(data.files):
                    raise ValueError('Incomplete page features')
                for name in required | {'merged_vision_features', 'pixel_values'}:
                    if not np.isfinite(data[name]).all():
                        raise ValueError(f'Non-finite {name}')
                np.testing.assert_allclose(query @ data['embeddings'].T,
                                           data['query_token_by_page_token'], atol=1e-5)
                regions = data['region_patch_overlap']
                profiles = data['region_query_maxsim']
                if regions.shape[1] != len(data['image_positions']) or len(regions) != len(profiles):
                    raise ValueError('Spatial alignment dimensions differ')
                empty = ~(regions > 0).any(axis=1)
                if not np.isnan(profiles[empty]).all() or not np.isfinite(profiles[~empty]).all():
                    raise ValueError('Invalid region profile missingness')
    print(json.dumps({'verified_cases': cases, 'verified_pages': 4 * cases, 'status': 'passed'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cases', type=int, default=17)
    args = parser.parse_args()
    verify(args.root, args.cases)
