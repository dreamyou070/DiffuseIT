from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments

def main(args) :
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_prompt()
    # image_editor.reconstruct_image()

if __name__ == "__main__":
    args = get_arguments()
    main(args)

