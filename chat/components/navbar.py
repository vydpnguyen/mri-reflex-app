import reflex as rx
from chat.state import State

def sidebar_chat(chat: str) -> rx.Component:
    """A sidebar chat item.

    Args:
        chat: The chat item.
    """
    return  rx.drawer.close(rx.hstack(
        rx.button(
            chat,
            on_click=lambda: State.set_chat(chat),
            width="80%", variant="surface",
            color_scheme="indigo",
            color="#3E63DD",
            background_color="#ffffff",
        ),
        rx.button(
            rx.icon(
                tag="trash",
                on_click=State.delete_chat,
                stroke_width=1,
            ),
            width="20%",
            variant="surface",
            color_scheme="red",
            background_color="#ffffff",
        ),
        width="100%",
    ))


def sidebar(trigger) -> rx.Component:
    """The sidebar component."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Chats", color="#3E63DD"),
                    rx.divider(),
                    rx.foreach(State.chat_titles, lambda chat: sidebar_chat(chat)),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color="#ffffff",
                outline="none",
            )
        ),
        direction="left",
    )


def modal(trigger) -> rx.Component:
    """A modal to create a new chat."""
    return rx.dialog.root(
        # wraps the control that will open the dialog
        rx.dialog.trigger(trigger),
        # contain the content of the dialog
        rx.dialog.content(
            rx.hstack(
                rx.input(
                    placeholder="Type something...",
                    # when the user clicks outside of a focused text input
                    on_blur=State.set_new_chat_name,
                    width=["15em", "20em", "30em", "30em", "30em", "30em"],
                ),
                # wrap the control that will close the dialog
                rx.dialog.close(
                    rx.button(
                        "Create chat",
                        on_click=State.create_chat,
                    ),
                ),
                background_color="ffffff",
                spacing="2",
                width="100%",
            ),
        ),
    )


def navbar():
    return rx.box(
        rx.hstack(
            # Avatar, heading and  Intros
            rx.hstack(
                rx.icon(
                    tag="heart-pulse",
                    color="#3E63DD",
                ),
                #rx.avatar(fallback="RC", variant="solid"),
                rx.heading("MRIBot", color="#3E63DD"),
                rx.desktop_only(
                    rx.badge(
                    # name of current chat
                    State.current_chat,
                    variant="solid",
                    background_color="#3E63DD"
                    )
                ),
                align_items="center",
            ),
            rx.hstack(
                # call function to create a modal dialog component
                modal(
                    rx.button(
                        rx.icon(
                                tag="message-square-plus",
                                color="#ffffff",
                            ),
                            background_color="#3E63DD",
                    ),
                ),
                # call function to create a sidebar
                sidebar(
                    rx.button(
                        rx.icon(
                            tag="messages-square",
                            color="#ffffff",
                        ),
                        background_color="#3E63DD",
                    )
                )
            ),
            justify_content="space-between",
            align_items="center",
            margin="0 50px",
        ),
        backdrop_filter="auto",
        backdrop_blur="lg",
        padding="12px",
        background_color="#ffffff",
        position="sticky",
        top="0",
        z_index="100",
        align_items="center",
    )
