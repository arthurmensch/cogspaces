import os

from sacred.observers import FileStorageObserver


class OurFileStorageObserver(FileStorageObserver):
    def artifact_event(self, name, filename):
        self.run_entry['artifacts'].append(name)
        self.save_json(self.run_entry, 'run.json')


def get_id(output_dir):
    if os.path.exists(output_dir):
        dir_nrs = [int(d) for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and
                   d.isdigit()]
        return max(dir_nrs + [0]) + 1
    else:
        return 0


def unroll_grid(grid):
    if 'grid' in grid:
        subgrids = grid.pop('grid')
        if not isinstance(subgrids, list):
            subgrids = [subgrids]
        subgrids = map(unroll_grid, subgrids)
        unrolled = []
        for subgrid in subgrids:
            for subsubgrid in subgrid:
                unrolled.append(dict(**grid, **subsubgrid))
        return unrolled
    else:
        return [grid]