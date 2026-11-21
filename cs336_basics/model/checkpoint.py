import torch
class checkpointing():
    def save_checkpoint(self,model, optimizer, iteration, out, extra=None):
        """
        Given a model, optimizer, and an iteration number, serialize them to disk.

        Args:
            model (torch.nn.Module): Serialize the state of this model.
            optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
            iteration (int): Serialize this value, which represents the number of training iterations
                we've completed.
            out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
            extra (dict | None): 续训需要的其他状态（随机数状态、已用时间等），存在 "extra" 键下。
        """
        obj = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "extra": extra or {},
            }
        torch.save(obj, out)

    def load_checkpoint(self, src, model, optimizer, map_location=None, return_extra=False):
        """
            Given a serialized checkpoint (path or file-like object), restore the
            serialized state to the given model and optimizer.
            Return the number of iterations that we previously serialized in
            the checkpoint.
        
            Args:
                src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
                model (torch.nn.Module): Restore the state of this model.
                optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
                map_location: 传给 torch.load。在 GPU 上存、在别的设备上读时需要（如 Colab 存的 checkpoint 拿到 Mac 上用）。
                return_extra (bool): True 时返回 (iteration, extra)。
            Returns:
                int: the previously-serialized number of iterations.
         """
        ckpt = torch.load(src, map_location=map_location)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if return_extra:
            return ckpt["iteration"], ckpt.get("extra", {})
        return ckpt["iteration"]