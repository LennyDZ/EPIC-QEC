from .qec_primitive import QECPrimitive


class CustomPrimitive(QECPrimitive):
    """A custom primitive that can be used to define new operations on the Tanner graph. This is a placeholder for users to implement their own primitives."""

    implementation_cls: type

    def get_implementation(self, registry) -> type:
        """Return the implementation class explicitly attached to this primitive."""
        return self.implementation_cls
