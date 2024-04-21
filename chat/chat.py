"""The main Chat app."""

import reflex as rx
from chat.components import chat, navbar
from chat.components.upload import upload_index

def index() -> rx.Component:
    """The main app."""
    return rx.chakra.vstack(
        navbar(),
        chat.chat(),
        chat.action_bar(),
        background_color="#E0F0F7",
        color=rx.color("indigo", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )


# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="indigo",
    ),
)
app.add_page(index)
