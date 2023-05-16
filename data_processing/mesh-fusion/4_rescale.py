import os
import trimesh
import argparse
import numpy as np

class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, default='./examples/3_simplify', help='Path to input directory.')
        parser.add_argument('--ref_dir', type=str, default='./examples/0_in', help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, default='./examples/4_rescale', help='Path to output directory; files within are overwritten!')
        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def run(self):
        """
        Run the tool, i.e. scale all found OFF files.
        """

        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)
        files = self.read_directory(self.options.in_dir)

        for filepath in files:
            # Get extents of model.
            # mesh = trimesh.load(filepath)
            # total_min = np.min(mesh.vertices, axis=0)
            # total_max = np.max(mesh.vertices, axis=0)
            # centers = (total_min + total_max)/2.
            
            # mesh_ref = trimesh.load(os.path.join(self.options.ref_dir, os.path.basename(filepath)))
            # min_ref = np.min(mesh_ref.vertices, axis=0)
            # max_ref = np.max(mesh_ref.vertices, axis=0)
            # centers_ref = (min_ref + max_ref)/2.

            # scale = np.linalg.norm(max_ref-min_ref)/np.linalg.norm(total_max-total_min)

            # mesh.vertices = (mesh.vertices-centers) * scale + centers_ref
            # mesh.export(os.path.join(self.options.out_dir, os.path.basename(filepath)))


            mesh_tmp = trimesh.load(filepath, process=False)

            vertice = mesh_tmp.vertices
            temp_max = vertice.max(0)
            temp_min = vertice.min(0)
            loc = (temp_max + temp_min) / 2
            scale = (temp_max - temp_min).max()

            # Transform input mesh
            mesh_tmp.apply_translation(-loc)
            mesh_tmp.apply_scale(1 / scale)
            mesh_tmp.export(os.path.join(self.options.out_dir, os.path.basename(filepath)))


if __name__ == '__main__':
    app = Scale()
    app.run()