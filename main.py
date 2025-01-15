import sys
import os
import json
import numpy as np
from PIL import Image
from yoloer import BoxDetection
from OCRENGINE import OCREngine
from dotenv import load_dotenv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, 
    QSplitter, QListWidget, QListWidgetItem, QFileDialog, 
    QSpinBox, QAbstractItemView, QListView, QComboBox, QPlainTextEdit
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QMouseEvent, QIcon, QFont, QKeySequence , QShortcut, QDesktopServices, QColor, QBrush
)
from PyQt6.QtCore import (
    Qt, QRect, QSize, QPoint, QUrl
)
load_dotenv()

engine_lists = ["Chinese", "Japanese"]
engine_from_env = os.getenv("OCR_ENGINE", "Chinese")
print(engine_from_env)

###############################################################################
# ImageLabel
###############################################################################
class ImageLabel(QLabel):
    """
    A QLabel for displaying an image and handling bounding boxes:
      - Store bounding boxes in a list of dictionaries:
        [
           {"id": <int>, "coords": (x, y, w, h)},
           ...
        ]
      - Left-click on box => highlight text (callback).
      - Right-click on box => select for deletion (press Delete to remove).
      - Drawing new boxes is done with left-click & drag.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = QPixmap()  # full-res
        self.display_pixmap = QPixmap()   # scaled for display
        self.scale_factor = 1.0

        self.drawing = False
        self.start_point = None
        self.end_point = None

        # bounding_boxes: list of dict: { "id": box_id, "coords": (x, y, w, h) }
        self.bounding_boxes = []
        self.selected_box_id = None  # which bounding box is selected for deletion
        self.selected_box_id_delete = None

        # External callbacks
        self.new_box_callback = None       # called when a new box is created
        self.left_click_box_callback = None  # called when left-click on existing box
        self.right_click_box_callback = None  # called when right-click on existing box

        self.next_box_id = 1  # to assign unique IDs to new boxes

        # Focus to catch keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.in_arrange_mode = False
        self.arrange_order = []

    def set_image(self, pixmap_full: QPixmap, scale_factor: float):
        """
        Setup the image label with a full pixmap and a scale factor.
        """
        self.original_pixmap = pixmap_full
        self.scale_factor = scale_factor

        w_disp = int(pixmap_full.width() * scale_factor)
        h_disp = int(pixmap_full.height() * scale_factor)
        self.display_pixmap = pixmap_full.scaled(
            w_disp, h_disp,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(self.display_pixmap)
        self.setFixedSize(self.display_pixmap.size())

    def _to_original_coords(self, rect: QRect):
        """
        Convert from display coords to original coords.
        """
        x = int(rect.x() / self.scale_factor)
        y = int(rect.y() / self.scale_factor)
        w = int(rect.width() / self.scale_factor)
        h = int(rect.height() / self.scale_factor)
        return (x, y, w, h)

    def _to_display_rect(self, x, y, w, h):
        """
        Convert original coords (x,y,w,h) => display rect.
        """
        return QRect(
            int(x * self.scale_factor),
            int(y * self.scale_factor),
            int(w * self.scale_factor),
            int(h * self.scale_factor),
        )

    def mousePressEvent(self, event: QMouseEvent):
        if self.display_pixmap.isNull():
            return
        clicked_point = event.position().toPoint()
        box_id = self._find_box_id_at_display_point(clicked_point)
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_box_id_delete = None
            if box_id is not None:
                # We clicked on an existing box
                if self.left_click_box_callback:
                    self.left_click_box_callback(box_id)
                    return
            # Begin drawing a new box
            self.drawing = True
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            # Attempt to select an existing bounding box
            if box_id is not None:
                self.selected_box_id_delete = box_id
                # Notify external
                if self.right_click_box_callback:
                    self.right_click_box_callback(box_id)
            else:
                self.selected_box_id_delete = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:
                # It's a valid bounding box
                coords = self._to_original_coords(rect)
                box_id = self.next_box_id
                self.next_box_id += 1

                self.bounding_boxes.append({
                    "id": box_id,
                    "coords": coords
                })
                self.update()

                # If we have a callback, pass the new box_id and coords
                if self.new_box_callback:
                    self.new_box_callback(box_id, coords)

            self.start_point = None
            self.end_point = None
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """
        interpret left double-click as 'highlight bounding box text.'
        Alternatively, can do single left-click highlight. 
        """
        """
        Placeholder for future use.
        """
        pass

    def _find_box_id_at_display_point(self, point: QPoint):
        """
        Return the box_id of the bounding box under 'point' (display coords),
        or None if none found. We'll look from front to back in self.bounding_boxes.
        """
        for box_dict in reversed(self.bounding_boxes):
            x, y, w, h = box_dict["coords"]
            r = self._to_display_rect(x, y, w, h)
            if r.contains(point):
                return box_dict["id"]
        return None

    def mouseReleaseEvent_withLeftClickHighlight(self, event: QMouseEvent):
        """
        - In mousePressEvent, detect if clicked on box
        - If yes, highlight text
        - If not, maybe we start drawing
        
        For now, we keep them separate: left-drag => new box, right-click => select for deletion.
        If you want a separate left-click for highlight, see code note below.
        """
        pass

    def left_click_highlight_check(self, point: QPoint):
        """
        If user left-clicked on an existing bounding box, we highlight the text.
        Called from mousePressEvent if desired. 
        We'll do it only if we are NOT starting a new box or something similar.
        """
        box_id = self._find_box_id_at_display_point(point)
        return box_id

    def keyPressEvent(self, event):
        """
        Press Delete => delete the selected box (right-click to select).
        After deleting, we call external so the text can be removed.
        """
        if event.key() == Qt.Key.Key_Delete:
            if self.selected_box_id_delete is not None:
                # Remove that bounding box
                removed_id = self.selected_box_id_delete
                self.bounding_boxes = [b for b in self.bounding_boxes if b["id"] != removed_id]
                self.selected_box_id_delete = None
                self.update()

                # If we have a callback to MainWindow, call it:
                if self.delete_callback:
                    self.delete_callback(removed_id)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        """_summary_
        Will be used to paint all the events, highlight boxes, Deletion boxes, Numbers when arranging.
        Args:
            event (_type_): _description_
        """
        super().paintEvent(event)
        if self.display_pixmap.isNull():
            return
    
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.display_pixmap)
    
        pen = pen_reset = QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
    
        # Draw bounding boxes
        for box_dict in self.bounding_boxes:
            box_id = box_dict["id"]
            x, y, w, h = box_dict["coords"]
            dr = self._to_display_rect(x, y, w, h)

            # Box Select Highlighter
            if box_id == self.selected_box_id:
                blue_pen = QPen(Qt.GlobalColor.blue, 5, Qt.PenStyle.SolidLine)  # highlight selected
                painter.setPen(blue_pen)

                blue_color = QColor(Qt.GlobalColor.blue)
                blue_color.setAlphaF(0.2)
                blue_brush = QBrush(blue_color)
                painter.setBrush(blue_brush)

                painter.drawRect(dr)
                #pen.setColor(Qt.GlobalColor.green)   # reset
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(pen_reset)
            else:
                painter.drawRect(dr)

            # Box Delete Highlighter
            if box_id == self.selected_box_id_delete:  # Right-click selected
                yellow_pen = QPen(Qt.GlobalColor.yellow, 5, Qt.PenStyle.SolidLine)
                painter.setPen(yellow_pen)

                yellow_color= QColor(Qt.GlobalColor.yellow)
                yellow_color.setAlphaF(0.2)
                yellow_brush = QBrush(yellow_color)
                painter.setBrush(yellow_brush)

                painter.drawRect(dr)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(pen)
            else:
                painter.drawRect(dr)

            # If in arrange mode, draw small numbering ***INSIDE THE LOOP***
            if self.in_arrange_mode and box_id in self.arrange_order:
                idx = self.arrange_order.index(box_id)
                box_number = idx + 1
                font1 = painter.font()
                font1.setPointSize(20) # Change Here
                painter.setFont(font1)
                painter.setPen(Qt.GlobalColor.red)
                #print(f"Drawing box_number={box_number} at box_id={box_id} in arrange mode") <- DEBUGGER
                painter.drawText(dr.topLeft() + QPoint(5, 15), str(box_number))
                pen.setColor(Qt.GlobalColor.green)
                painter.setPen(pen)
    
        # If user is drawing a new rectangle
        if self.drawing and self.start_point and self.end_point:
            pen.setColor(Qt.GlobalColor.red)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            draw_rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(draw_rect)

###############################################################################
# MainWindow
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Setsu Scans TLer Helper - By Satyam")
        icon_path = os.path.join(os.path.dirname(__file__), 'icon', 'logo.png')
        self.setWindowIcon(QIcon(icon_path))
        self.showMaximized()
        self.in_arrange_mode = False
        self.arrange_order = []

        self.image_directory = ""
        self.image_files = []
        self.current_image_index = -1

        self.ocr_engine = None
        # For each image, we store a list of bounding_box dicts, each with:
        # { "id": box_id, "coords": (x, y, w, h), "lines": [line1, line2, ...], "user_texts": [...] }
        self.boxes_data = {}
        # Because YOLO or manual drawing can create new boxes with text, etc.

        ############################
        # Main Layout
        ############################
        # We want 3 main sections horizontally:
        #  [ Left: Buttons |  Center: Image  | Right: Vertical: (TextList on top, Thumbnails below) ]
        main_widget = QWidget()
        main_hlayout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # === Left Panel: Buttons / Controls ===
        left_panel = QWidget()
        left_vlayout = QVBoxLayout(left_panel)

        # Normal Usage
        self.open_button = QPushButton("Open Directory")
        self.open_button.clicked.connect(self.open_directory)
        left_vlayout.addWidget(self.open_button)

        self.prev_button = QPushButton("Previous Image")
        self.prev_button.clicked.connect(self.prev_image)
        left_vlayout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.next_image)
        left_vlayout.addWidget(self.next_button)

        self.yolo_button = QPushButton("OCR This Image")
        self.yolo_button.clicked.connect(self.perform_yolo_ocr)
        left_vlayout.addWidget(self.yolo_button)

        self.yolo_all_button = QPushButton("OCR All Images")
        self.yolo_all_button.clicked.connect(self.perform_yolo_all_images)
        left_vlayout.addWidget(self.yolo_all_button)


        self.clear_button = QPushButton("Clear All (This Image)")
        self.clear_button.clicked.connect(self.clear_all)
        left_vlayout.addWidget(self.clear_button)
        
        # Arrange the Boxes
        self.arrange_mode_on_button = QPushButton("Arrange Mode ON")
        self.arrange_mode_on_button.clicked.connect(self.enable_arrange_mode)
        left_vlayout.addWidget(self.arrange_mode_on_button)

        self.arrange_mode_off_button = QPushButton("Arrange Mode OFF")
        self.arrange_mode_off_button.clicked.connect(self.disable_arrange_mode)
        left_vlayout.addWidget(self.arrange_mode_off_button)


        self.arrange_button = QPushButton("Auto Arrange (Placeholder)")
        self.arrange_button.clicked.connect(self.on_arrange_button)
        left_vlayout.addWidget(self.arrange_button)

        # Font size spin
        font_label = QLabel("Text Size:")
        left_vlayout.addWidget(font_label)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 72)
        self.font_spin.setValue(12)
        self.font_spin.valueChanged.connect(self.change_text_size)
        left_vlayout.addWidget(self.font_spin)

        # Save Annotations
        self.update_button = QPushButton("Save File")
        self.update_button.clicked.connect(self.update_annotations_file)
        left_vlayout.addWidget(self.update_button)

        # Export RAW
        self.export_raw_button = QPushButton("Export RAW (RAW.txt)")
        self.export_raw_button.clicked.connect(self.export_raw_text)
        left_vlayout.addWidget(self.export_raw_button)
        
        # Export TL
        self.export_tl_button = QPushButton("Export TL (TL.txt)")
        self.export_tl_button.clicked.connect(self.export_user_text)
        left_vlayout.addWidget(self.export_tl_button)
        
        # Repopulate Text
        self.repopulate_button = QPushButton("Repopulate Texts")
        self.repopulate_button.clicked.connect(self.repopulate_texts)
        left_vlayout.addWidget(self.repopulate_button)

        left_vlayout.addStretch(1)  # push items up
        # Add left panel to main layout
        main_hlayout.addWidget(left_panel, stretch=1)

        # === Center Panel: Image in ScrollArea ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        self.scroll_area = QScrollArea()
        self.image_label = ImageLabel()
        self.image_label.new_box_callback = self.on_new_box_created
        self.image_label.right_click_box_callback = self.on_right_click_box
        self.image_label.left_click_box_callback = self.on_left_click_box
        self.image_label.delete_callback = self.on_box_deleted
        # If you want single left-click to highlight text, you could also set:
        # self.image_label.left_click_box_callback = self.on_left_click_box
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        center_layout.addWidget(self.scroll_area)

        main_hlayout.addWidget(center_panel, stretch=4)

        # === Right Panel: text list on top, thumbnails on bottom ===
        right_panel = QSplitter(Qt.Orientation.Vertical)
        # 1) Text list
        self.text_list = QListWidget()

        self.text_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.text_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.text_list.model().rowsMoved.connect(self.on_text_list_reordered)
        self.text_list.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.text_list.setStyleSheet("""
          QListWidget::item {
              border-bottom: 1px solid #AAAAAA;
              margin-bottom: 2px;
              /* Optional: ensure a transparent or default background in normal state */
              background: transparent; 
          }
          /* When an item is selected, give it a visible highlight color */
          QListWidget::item:selected {
              background-color: #6FA6E6; /* or any color you like */
              color: white;             /* text color on selection */
          }
          /* Active vs. inactive states can be styled too, if needed: */
          QListWidget::item:selected:active {
              background-color: #6FA6E6;
              color: white;
          }
          QListWidget::item:selected:!active {
              background-color: #A7C7E7;
              color: black;
          }
      """)
        self.text_list.itemSelectionChanged.connect(self.on_text_list_selection_changed)

        # User Text list
        self.user_text_list = QListWidget()
        self.user_text_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.user_text_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.user_text_list.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.user_text_list.itemSelectionChanged.connect(self.on_user_text_list_selection_changed)
        self.user_text_list.setStyleSheet("""
          QListWidget::item {
              border-bottom: 1px solid #AAAAAA;
              margin-bottom: 2px;
              /* Optional: ensure a transparent or default background in normal state */
              background: transparent; 
          }
          /* When an item is selected, give it a visible highlight color */
          QListWidget::item:selected {
              background-color: #6FA6E6; /* or any color you like */
              color: white;             /* text color on selection */
          }
          /* Active vs. inactive states can be styled too, if needed: */
          QListWidget::item:selected:active {
              background-color: #6FA6E6;
              color: white;
          }
          QListWidget::item:selected:!active {
              background-color: #A7C7E7;
              color: black;
          }
      """)

        # 2) Thumbnails
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListView.ViewMode.IconMode)
        self.thumbnail_list.setWrapping(True) #Wrapping
        self.thumbnail_list.setFlow(QListWidget.Flow.LeftToRight)
        self.thumbnail_list.setGridSize(QSize(120, 120))
        self.thumbnail_list.setIconSize(QSize(100, 100))
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.itemClicked.connect(self.thumbnail_clicked)

        right_panel.addWidget(self.text_list)      # top
        right_panel.addWidget(self.user_text_list) # middle
        right_panel.addWidget(self.thumbnail_list) # bottom
        right_panel.setStretchFactor(0, 1)
        right_panel.setStretchFactor(0, 1)
        right_panel.setStretchFactor(1, 1)

        main_hlayout.addWidget(right_panel, stretch=2)

        self.all_buttons = [
            self.open_button,
            self.prev_button,
            self.next_button,
            self.yolo_button,
            self.clear_button,
            self.arrange_mode_on_button,
            self.update_button,          
            self.export_raw_button,   
            self.yolo_all_button
        ]

        # Extra Section
        self.copyright_button = QPushButton("By Satyam, Made for Setsu Scans")
        self.issues_button = QPushButton("Submit Issue")
        self.github_button = QPushButton("Github")

        

        for btn_x in [self.copyright_button, self.issues_button, self.github_button]:
            btn_x.setStyleSheet("""
                QPushButton {
                    color: black;
                    text-decoration: underline;
                    background: white;
                    border: 2px;
                    padding: 5px 0;
                    font-size: 14px;
                }
                QPushButton:hover {
                    color: #8129ff;
                }
            """)
            btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        self.issues_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/yourrepo/issues")))
        self.github_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/yourrepo")))
        left_vlayout.addWidget(self.copyright_button)
        left_vlayout.addWidget(self.issues_button)
        left_vlayout.addWidget(self.github_button)

        self.engine_selector = QComboBox()
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        #self.log_console.setMaximumBlockCount(3)
        self.log_console.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_console.setFixedHeight(60)
        left_vlayout.addWidget(self.log_console)
        for eng in engine_lists:
            self.engine_selector.addItem(eng)
        if engine_from_env in engine_lists:
            self.engine_selector.setCurrentText(engine_from_env)
        else:
            self.engine_selector.setCurrentIndex(0)
        self.engine_selector.currentTextChanged.connect(self.on_engine_changed)
        left_vlayout.addWidget(self.engine_selector)
        default_engine_name = self.engine_selector.currentText()
        self.ocr_engine = OCREngine(default_engine_name)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_directory)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.next_image)
        QShortcut(QKeySequence("D"), self, self.next_image)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence("A"), self, self.prev_image)
        QShortcut(QKeySequence("W"), self, self.perform_yolo_ocr)
        QShortcut(QKeySequence("Ctrl+S"), self, self.update_annotations_file)
        self.arrange_button_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        self.arrange_button_shortcut.activated.connect(self.enable_arrange_mode)
        self.noarrange_button_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        self.noarrange_button_shortcut.activated.connect(self.disable_arrange_mode)
        self.arrange_button_shortcut.setEnabled(True)   # user can press Tab to enable
        self.noarrange_button_shortcut.setEnabled(False)

    ########################################################################
    # DEBUG
    ########################################################################
    def print_items(self):
        for i in range(self.text_list.count()):
            print(self.text_list.item(i).text())
            print(self.text_list.item(i).data(Qt.ItemDataRole.UserRole))
    ########################################################################
    # Change Engine
    ########################################################################

    def on_engine_changed(self, engine_name):
      """
      Called whenever the user selects a new engine from the drop-down.
      We instantiate a new OCREngine so future OCR calls go to the new engine.
      """
      print(f"Selected engine: {engine_name}")
      old_engine = self.ocr_engine
      self.ocr_engine = OCREngine(engine_name)
      if old_engine is not None:
        old_engine.cleanup()
        del old_engine

    ########################################################################
    # Loading/Saving Images & Annotations
    ########################################################################
    def log(self, message: str):
        """
        Append a line to the log_console. The oldest lines will disappear
        if we exceed 3 lines because we used setMaximumBlockCount(3).
        """
        self.log_console.appendPlainText(message)

    def on_box_deleted(self, removed_id):
      # 1) Gather current UI data
      self.update_in_memory_annotations()

      # 2) Remove from boxes_data
      if 0 <= self.current_image_index < len(self.image_files):
          file_path = self.image_files[self.current_image_index]
          old_data = self.boxes_data.get(file_path, [])
          new_data = [d for d in old_data if d["id"] != removed_id]
          self.boxes_data[file_path] = new_data

          # 3) Re-populate text fields
          self.populate_text_list(new_data, "Text Bubble Deleted")

          # 4) Optionally save
          # self.save_current_annotations()

    def repopulate_texts(self):
        """
        Discard any unsaved changes in the text widgets and reload them
        from self.boxes_data for the current image.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            print("No valid image loaded.")
            return
        file_path = self.image_files[self.current_image_index]

        # 1) Get the current image data from memory
        file_data = self.boxes_data.get(file_path, [])
        # 2) Make update to the memory
        self.update_in_memory_annotations()
        # 3) Repopulate the text widgets
        self.populate_text_list(file_data, "Repopulated Text Boxes")
        #print("Repopulated texts from in-memory data.")

    def update_in_memory_annotations(self):
        """
        Gather the current bounding boxes & text from the UI
        and store them in self.boxes_data for the active image.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return
        file_path = self.image_files[self.current_image_index]
        file_data = self.gather_file_data_from_ui()  # same as in save_current_annotations
        self.boxes_data[file_path] = file_data

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if not directory:
            return
        self.image_directory = directory
        self.image_files = []
        for f in os.listdir(directory):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_files.append(os.path.join(directory, f))
        self.image_files.sort()
        self.current_image_index = 0

        # Load JSON if it exists
        json_path = os.path.join(directory, "annotations.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                # data: { file_path: [ { "id":..., "coords":..., "lines":[...] }, ... ], ... }
                self.boxes_data = data
        else:
            self.boxes_data = {}

        self.load_image()
        self.load_thumbnails()

    def load_image(self):
        if not (0 <= self.current_image_index < len(self.image_files)):
            return
        
        file_path = self.image_files[self.current_image_index]
        file_data = self.boxes_data.get(file_path, [])
        self.populate_text_list(file_data, f"{file_path}")
        
        pix = QPixmap(file_path)
        if pix.isNull():
            return

        # Scale factor if bigger than screen
        screen_geo = QApplication.primaryScreen().geometry()
        sw, sh = screen_geo.width(), screen_geo.height()
        iw, ih = pix.width(), pix.height()

        scale_factor = 1.0
        if iw > sw or ih > sh:
            scale_w = sw / iw
            scale_h = sh / ih
            scale_factor = min(scale_w, scale_h)

        self.image_label.set_image(pix, scale_factor)
        # Clear existing bounding boxes from the label
        self.image_label.bounding_boxes.clear()

        # If we have data for this file
        file_data = self.boxes_data.get(file_path, [])
        # Example: [ { "id":..., "coords":(x,y,w,h), "lines":["...","..."] }, ... ]
        # Populate bounding boxes
        for box_info in file_data:
            self.image_label.bounding_boxes.append({
                "id": box_info["id"],
                "coords": tuple(box_info["coords"])  # (x, y, w, h)
            })
            # track the highest box_id so we continue from there
            if box_info["id"] >= self.image_label.next_box_id:
                self.image_label.next_box_id = box_info["id"] + 1
        self.image_label.bounding_boxes = self.arrange_bounding_box_sequence(
            self.image_label.bounding_boxes
        )
        # Load text list
        self.populate_text_list(file_data)

        # Update thumbnail highlight
        self.thumbnail_list.setCurrentRow(self.current_image_index)

    def populate_text_list(self, file_data, intent = None):
        """
        Clear text_list and re-populate from file_data (per-box).
        We store each line as a separate QListWidgetItem with bounding_box_id in .UserRole
        This is a main Function along with some others for memory management and proper rendering of texts on the screen.
        """
        if intent: # <- DEBUGGER
            #print(intent)
            self.log(intent)
        self.text_list.clear()
        self.user_text_list.clear()
        #print("[DEBUG] populate_text_list CALLED. file_data length:", len(file_data))
        for box_info in file_data:
            box_id = box_info["id"]
            # print(f"[DEBUG]   Box {box_id}")
            for line in box_info["lines"]:
                item = QListWidgetItem(line)
                # store bounding box id
                item.setData(Qt.ItemDataRole.UserRole, box_id)
                # allow editing
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.text_list.addItem(item)

            user_lines = box_info.get("user_lines", [])
            # If user_lines is empty, insert a placeholder
            if not user_lines:
                placeholder = self.get_user_placeholder_text(box_info)
                user_lines = [placeholder]
                box_info["user_lines"] = user_lines

            for line in user_lines:
                # print(f"[DEBUG]     -> adding user text: {line}")  # <-- debugging
                item = QListWidgetItem(line)
                item.setData(Qt.ItemDataRole.UserRole, box_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.user_text_list.addItem(item)

    def get_user_placeholder_text(self, box_info):
        """_summary_
        Currently a placeholder, Can also be used to include a Translator to translate the OCR Text to english and send back the text.
        Args:
            box_info (_list_): contains any other language text in a list of string, use as box_info["lines"] to get the lines

        Returns:
            _type_: _description_
        """
        box_id = box_info["id"]
        return f"Box {box_id}"
    
    def gather_file_data_from_ui(self):
        """
        Gather the bounding boxes & lines from the UI for the current image,
        and return a list of box-info dict:
           { "id": box_id, 
           "coords":(x,y,w,h), 
           "lines":[...], 
           "user_lines": [...]}
        We'll read the bounding boxes from self.image_label, 
        then read the text lines from self.text_list (group by box_id).
        """
        file_data = []
        # Create a dict from box_id => (x,y,w,h)
        box_dict_map = {}
        for box_dict in self.image_label.bounding_boxes:
            box_id = box_dict["id"]
            box_dict_map[box_id] = box_dict["coords"]

        # Create a dict from box_id => list_of_lines
        lines_map = {}
        for i in range(self.text_list.count()):
            item = self.text_list.item(i)
            line = item.text()
            box_id = item.data(Qt.ItemDataRole.UserRole)
            if box_id is not None:
                if box_id not in lines_map:
                    lines_map[box_id] = []
                lines_map[box_id].append(line)

        user_lines_map = {}
        for i in range(self.user_text_list.count()):
            item = self.user_text_list.item(i)
            line = item.text()
            b_id = item.data(Qt.ItemDataRole.UserRole)
            if b_id not in user_lines_map:
                user_lines_map[b_id] = []
            user_lines_map[b_id].append(line)

        # Combine
        for box_id, coords in box_dict_map.items():
            lines = lines_map.get(box_id, [])
            user_lines = user_lines_map.get(box_id, [])
            file_data.append({
                "id": box_id,
                "coords": coords,
                "lines": lines,
                "user_lines": user_lines,
            })
        return file_data
    
    def update_annotations_file(self):
        """
        Explicitly writes the current annotations to the JSON file.
        """
        self.log("File Saved")
        self.save_current_annotations()
        # Optionally print a message or show a message box:
        # print("Annotations file updated successfully.")

    def save_current_annotations(self):
        """
        Loads the existing annotations.json (if any),
        updates ONLY the current image data,
        then re-writes the entire file so we don't end up with duplications.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return

        file_path = self.image_files[self.current_image_index]
        file_data = self.gather_file_data_from_ui()  # your custom function
        # file_data = [ { "id":..., "coords":..., "lines": [...] }, ... ]

        if self.image_directory:
            json_path = os.path.join(self.image_directory, "annotations.json")

            # 1) Load existing data if file exists
            if os.path.exists(json_path):
                with open(json_path, "r", encoding='utf-8') as f:
                    overall_data = json.load(f)
            else:
                overall_data = {}

            # 2) Update the data for this specific file_path
            overall_data[file_path] = file_data

            # 3) Rewrite the entire file
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(overall_data, f, indent=4, ensure_ascii=False)

    def load_thumbnails(self):
        self.thumbnail_list.clear()
        for path in self.image_files:
            item = QListWidgetItem()
            pix = QPixmap(path).scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            item.setIcon(QIcon(pix))
            item.setData(Qt.ItemDataRole.UserRole, path)
            # optionally set text
            item.setText(os.path.basename(path))
            self.thumbnail_list.addItem(item)
        if 0 <= self.current_image_index < len(self.image_files):
            self.thumbnail_list.setCurrentRow(self.current_image_index)

    def thumbnail_clicked(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path in self.image_files:
            self.update_in_memory_annotations()
            self.save_current_annotations()
            new_index = self.image_files.index(file_path)
            self.current_image_index = new_index
            self.load_image()

    ########################################################################
    # Navigation
    ########################################################################
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.update_in_memory_annotations()
            self.save_current_annotations()
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_image_index > 0:
            self.update_in_memory_annotations()
            self.save_current_annotations()
            self.current_image_index -= 1
            self.load_image()

    ########################################################################
    # Callbacks from ImageLabel
    ########################################################################
    def on_text_list_reordered(self):
        """
        Called when the user finishes dragging in self.user_text_list.
        We read the new item order, rebuild a 'file_data' that matches it,
        then we store file_data in self.boxes_data[file_path],
        and repopulate so the UI is consistent with memory.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return

        file_path = self.image_files[self.current_image_index]
        # The original data (list of bounding boxes)
        old_file_data = self.boxes_data.get(file_path, [])

        # 1) Read the new order from the user_text_list
        # We'll build a dict from box_id => list_of_user_lines
        new_user_map = {}
        for i in range(self.user_text_list.count()):
            item = self.user_text_list.item(i)
            line = item.text()
            box_id = item.data(Qt.ItemDataRole.UserRole)
            if box_id not in new_user_map:
                new_user_map[box_id] = []
            new_user_map[box_id].append(line)

        # 2) Now build a *new* file_data that has the same bounding boxes
        # but with 'user_lines' in the newly dragged order. Typically,
        # you only reorder lines within the same box_id, or you might reorder boxes themselves.
        # We'll assume here you're just reordering lines *within* the same box ID.

        # If you want to reorder entire boxes, you'd track each item as one "box".
        # But if you want to reorder lines *within a single box*, you'd need a different approach.
        # Let's assume each item = one line belonging to the box_id => so each box can have multiple lines.

        new_file_data = []
        for box_info in old_file_data:
            b_id = box_info["id"]
            # lines remain the same
            box_lines = box_info.get("lines", [])
            # user_lines become the new order from new_user_map
            new_user_lines = new_user_map.get(b_id, [])
            new_file_data.append({
                "id": b_id,
                "coords": box_info["coords"],
                "lines": box_lines,
                "user_lines": new_user_lines
            })

        # 3) Overwrite the old data with the new
        self.boxes_data[file_path] = new_file_data

        # 4) (Optional) Re-populate to ensure UI matches the final data
        #    If you trust that the 'QListWidget' is already showing the correct final order,
        #    you might skip this. But to ensure no mismatch, you can do:
        self.populate_text_list(new_file_data)

    def on_text_list_selection_changed(self):
        """
        Called when the selection in self.text_list changes.
        We'll highlight the corresponding bounding box in blue
        and also select items in the other text list (user_text_list).
        """
        selected_items = self.text_list.selectedItems()
        if not selected_items:
            # If nothing is selected, maybe reset or do nothing
            self.image_label.selected_box_id = None
            self.image_label.update()
            return

        # For simplicity, let's consider the first selected item only
        item = selected_items[0]
        box_id = item.data(Qt.ItemDataRole.UserRole)

        # 1) Highlight the bounding box
        self.image_label.selected_box_id = box_id
        self.image_label.update()

        # 2) Also highlight the same box_id in the user_text_list
        self.user_text_list.blockSignals(True)  # avoid infinite loop
        self.user_text_list.clearSelection()
        for i in range(self.user_text_list.count()):
            u_item = self.user_text_list.item(i)
            if u_item.data(Qt.ItemDataRole.UserRole) == box_id:
                u_item.setSelected(True)
        self.user_text_list.blockSignals(False)

    def on_text_list_reordered(self):
        """
        After user drags items in self.text_list, we read the new order and
        update self.boxes_data so it reflects the new line order.
        """
        self.log("on_text_list_reordered called.")
    
        if not (0 <= self.current_image_index < len(self.image_files)):
            self.log("No valid image loaded.")
            return

        file_path = self.image_files[self.current_image_index]
        file_data = self.boxes_data.get(file_path, [])

        # Create a mapping from box_id to box_info for quick access
        box_map = {box['id']: box for box in file_data}

        reordered_data = []

        # Iterate over the QListWidget items in their new order
        for i in range(self.text_list.count()):
            item = self.text_list.item(i)
            box_id = item.data(Qt.ItemDataRole.UserRole)

            if box_id in box_map:
                reordered_data.append(box_map[box_id])
            else:
                self.log(f"Warning: box_id {box_id} not found in box_map.")

        # Update self.boxes_data with the reordered list
        self.boxes_data[file_path] = reordered_data

        # Optionally update ImageLabel's bounding_boxes if necessary
        self.image_label.bounding_boxes = [
            {"id": box["id"], "coords": box["coords"]}
            for box in reordered_data
        ]

        # Re-populate both text lists to reflect the new order
        self.populate_text_list(reordered_data)
        self.update_annotations_file()
        self.log("boxes_data updated with new order from text_list.")

    def on_user_text_list_reordered(self): # Don't Add functionaly for this rn
        """
        After user drags items in self.user_text_list, we read the new order and
        update self.boxes_data so it reflects the new user_text order.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return

        file_path = self.image_files[self.current_image_index]
        file_data = self.boxes_data.get(file_path, [])

        new_user_map = {}
        for i in range(self.user_text_list.count()):
            item = self.user_text_list.item(i)
            line = item.text()
            box_id = item.data(Qt.ItemDataRole.UserRole)
            if box_id not in new_user_map:
                new_user_map[box_id] = []
            new_user_map[box_id].append(line)

        for box_info in file_data:
            box_id = box_info["id"]
            if box_id in new_user_map:
                box_info["user_lines"] = new_user_map[box_id]

        self.boxes_data[file_path] = file_data
        # optional: self.populate_text_list(file_data)

    def on_user_text_list_selection_changed(self):
        """
        Called when the selection in self.user_text_list changes.
        We do the same logic but in reverse (highlight box + text_list).
        """
        selected_items = self.user_text_list.selectedItems()
        if not selected_items:
            self.image_label.selected_box_id = None
            self.image_label.update()
            return

        item = selected_items[0]
        box_id = item.data(Qt.ItemDataRole.UserRole)

        # 1) Highlight bounding box
        self.image_label.selected_box_id = box_id
        self.image_label.update()

        # 2) Also highlight the same box_id in text_list
        self.text_list.blockSignals(True)
        self.text_list.clearSelection()
        for i in range(self.text_list.count()):
            o_item = self.text_list.item(i)
            if o_item.data(Qt.ItemDataRole.UserRole) == box_id:
                o_item.setSelected(True)
        self.text_list.blockSignals(False)

    def on_left_click_box(self, box_id):
      """
      If we're in arrange mode, record the click order (box_id).
      Otherwise, just highlight text lines.
      """
      if self.in_arrange_mode:
          # If this box_id wasn't already clicked, add it to the sequence
          if box_id in self.arrange_order:
            self.arrange_order.remove(box_id)

          if box_id not in self.arrange_order:
              self.arrange_order.append(box_id)
          # No need to highlight text, so return
          self.image_label.update()
          return
      
      """
      Highlight text lines in self.text_list that belong to the bounding box with 'box_id'.
      """
      # 1) Clear all selection in the text_list
      self.text_list.clearSelection()
      # 2) Iterate through items, select those that match box_id
      for i in range(self.text_list.count()):
          item = self.text_list.item(i)
          item_box_id = item.data(Qt.ItemDataRole.UserRole)
          if item_box_id == box_id:
              item.setSelected(True)
      self.user_text_list.clearSelection()
      for i in range(self.user_text_list.count()):
          item = self.user_text_list.item(i)
          if item.data(Qt.ItemDataRole.UserRole) == box_id:
              item.setSelected(True)

    def on_new_box_created(self, box_id, coords):
      """
      Called right after a new bounding box is drawn.
      We'll run OCR on the new region, store it in self.boxes_data
      with an empty 'user_lines' field, and re-populate the UI 
      so the new box is immediately visible in both text widgets.
      """
      if not (0 <= self.current_image_index < len(self.image_files)):
          return
      file_path = self.image_files[self.current_image_index]

      # 1) Ensure we don't lose any current UI edits
      self.update_in_memory_annotations()  # <-- ADDED

      x, y, w, h = coords
      if w <= 0 or h <= 0:
          return

      # 2) Crop the region from the original image for OCR
      pil_img = Image.open(file_path)
      roi = pil_img.crop((x, y, x + w, y + h))

      # 3) Run OCR with the chosen engine
      results = self.ocr_engine.predict(np.array(roi))  # <-- CHANGED

      # 4) Create or retrieve the existing data for this image
      file_data = self.boxes_data.get(file_path, [])

      # 5) Build a new dict for the newly drawn box
      new_box_dict = {
          "id": box_id,
          "coords": (x, y, w, h),
          "lines": results if results else [],
          "user_lines": []  # placeholder list for user text
      }
      file_data = [d for d in file_data if d["id"] != box_id]
      # 6) Append the new box dict to the existing data
      updated_data = file_data + [new_box_dict]
      self.boxes_data[file_path] = updated_data

      # 7) Update the ImageLabel's bounding_boxes (so it can draw the new box)
      self.image_label.bounding_boxes.clear()
      for box_info in updated_data:
          self.image_label.bounding_boxes.append({
              "id": box_info["id"],
              "coords": box_info["coords"]
          })

      # 8) Re-populate both text widgets (OCR + user text) with updated_data
      self.populate_text_list(updated_data, "new_box_created")  # <-- ADDED

      # 9) Finally, update the label so the new box appears immediately
      self.image_label.update()

    def on_right_click_box(self, box_id, delete=False):
        """
        Called when user right-clicks a bounding box (to select it),
        OR after pressing Delete, we call it again with delete=True 
        to remove that box's text items.
        """
        if delete:
            # remove all items from text_list with this box_id
            i = 0
            while i < self.text_list.count():
                item = self.text_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == box_id:
                    self.text_list.takeItem(i)  # remove
                else:
                    i += 1
        else:
            # The user just right-clicked to select. 
            # If you want to highlight the text or something, do it here.
            pass

    ########################################################################
    # Clear / Font Size
    ########################################################################
    def clear_all(self):
        """
        Remove all bounding boxes and all text from the current image only.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return
        self.image_label.bounding_boxes.clear()
        self.text_list.clear()
        self.user_text_list.clear()
        self.image_label.update()

    def change_text_size(self):
        size = self.font_spin.value()
        font = QFont()
        font.setPointSize(size)
        self.text_list.setFont(font)
        self.user_text_list.setFont(font)

    ########################################################################
    # YOLO
    ########################################################################

    def perform_yolo_all_images(self):
        """
        Runs YOLO detection + OCR on *all* images in self.image_files.
        Updates self.boxes_data (and the image_label bounding_boxes for the current image, if relevant),
        then saves the updated annotations.
        """
        if not self.image_files:
            print("No images loaded.")
            return

        detector = BoxDetection()

        for i, file_path in enumerate(self.image_files, start=1):
            #print(f"[{i}/{len(self.image_files)}] Running YOLO on: {file_path}")

            # 1) Perform YOLO detection
            yolo_boxes = detector.predict(file_path)  # returns [(x1, y1, x2, y2), ...]

            # 2) Convert YOLO boxes into the format your code expects
            #    i.e. { "id": <box_id>, "coords": (x, y, w, h), "lines": [...] }
            #    We'll create a list of new boxes (for YOLO) to append
            pil_img = Image.open(file_path)
            img_np = np.array(pil_img)

            # If the file_path has existing data, load it
            existing_data = self.boxes_data.get(file_path, [])
            # existing_box_ids = {d["id"] for d in existing_data}
            # We might track next_box_id by scanning existing_data or using the global self.image_label.next_box_id
            # But for a quick approach, let's find a safe max_id from existing_data:
            max_id = 0
            for d in existing_data:
                if d["id"] > max_id:
                    max_id = d["id"]

            new_boxes = []
            for (x1, y1, x2, y2) in yolo_boxes:
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                max_id += 1
                box_id = max_id

                # OCR on the cropped region
                cropped = img_np[y1:y2, x1:x2]
                results = self.ocr_engine.predict(cropped)  # or self.reader.readtext if not using engine
                new_boxes.append({
                    "id": box_id,
                    "coords": (x1, y1, w, h),
                    "lines": results if results else [],
                    "user_lines": []
                })

            # 3) Append the new boxes to existing data
            updated_data = existing_data + new_boxes
            self.boxes_data[file_path] = updated_data

        # 4) Finally, save annotations so changes are written to JSON
        self.save_current_annotations()
        #print("YOLO All Images completed and annotations saved.")
        current_file = self.image_files[self.current_image_index]
        self.load_image()

    def perform_yolo_ocr(self):
        """_summary_
        Performs YOLO Bubble Detection.
        Major Issue: There's a scenario where it detects same bubble multiple times with the exact bouding box coords.
        """
        if not (0 <= self.current_image_index < len(self.image_files)):
            return

        # 1) Clear the old bounding boxes/text from the current image
        self.clear_all()

        file_path = self.image_files[self.current_image_index]
        detector = BoxDetection()
        yolo_boxes = detector.predict(file_path)  # each is (x1, y1, x2, y2)

        pil_img = Image.open(file_path)
        img_np = np.array(pil_img)

        # 2) Build a single new_data list for all YOLO boxes
        new_data = []
        current_box_id = self.image_label.next_box_id
        #print("loop in")
        for (x1, y1, x2, y2) in yolo_boxes:
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            box_id = current_box_id
            current_box_id += 1

            # OCR for this bounding box
            cropped = img_np[y1:y2, x1:x2]
            results = self.ocr_engine.predict(cropped)

            # Add to new_data
            new_data.append({
                "id": box_id,
                "coords": (x1, y1, w, h),
                "lines": results if results else [],
                "user_lines": []
            })
        #print("loop out")
        # 3) Update next_box_id once after processing all boxes
        self.image_label.next_box_id = current_box_id

        # 4) (Optional) Reorder them if you have custom logic
        new_data = self.arrange_bounding_box_sequence(new_data)

        # 5) Update self.boxes_data for this file
        self.boxes_data[file_path] = new_data

        # 6) Set bounding boxes in the ImageLabel
        self.image_label.bounding_boxes.clear()
        for box_dict in new_data:
            self.image_label.bounding_boxes.append({
                "id": box_dict["id"],
                "coords": box_dict["coords"]
            })

        # 7) Finally, populate the text widgets once
        self.populate_text_list(new_data, "Text Bubble Detection Done.")

        # 8) Update the image label
        self.image_label.update()


#    def perform_yolo_ocr(self):
#        if not (0 <= self.current_image_index < len(self.image_files)):
#            return
#        self.clear_all()
#        file_path = self.image_files[self.current_image_index]
#        detector = BoxDetection()
#        yolo_boxes = detector.predict(file_path)  # each is (x1, y1, x2, y2)
#
#        # For each YOLO box, add bounding box + do OCR
#        pil_img = Image.open(file_path)
#        img_np = np.array(pil_img)
#        
#        new_data = []
#        while_simplify_id = self.image_label.next_box_id
#
#        for (x1, y1, x2, y2) in yolo_boxes:
#            w, h = x2 - x1, y2 - y1
#            if w <= 0 or h <= 0:
#                continue
#
#            # Assign a new box_id
#            box_id = while_simplify_id
#            while_simplify_id += 1
#
#            # Add to self.image_label.bounding_boxes
#            self.image_label.bounding_boxes.append({
#                "id": box_id,
#                "coords": (x1, y1, w, h)
#            })
#
#            # OCR
#            cropped = img_np[y1:y2, x1:x2]
#            #results = self.reader.readtext(cropped, detail=0, paragraph=False)
#            results = self.ocr_engine.predict(cropped)
#            new_data.append({
#                "id": box_id,
#                "coords": (x1, y1, w, h),
#                "lines": results if results else [],
#                "user_lines": []  # <-- ADDED user_lines so we can store user text
#            })
#
#        self.image_label.next_box_id = while_simplify_id
#        new_data = self.arrange_bounding_box_sequence(new_data)
#        self.boxes_data[file_path] = new_data
#        self.image_label.bounding_boxes.clear()
#        for box_dict in new_data:
#            self.image_label.bounding_boxes.append({
#                "id": box_dict["id"],
#                "coords": box_dict["coords"]
#            })
#        self.populate_text_list(new_data)
#        self.image_label.update()

    ########################################################################
    # Arrange Modes
    ########################################################################

    def enable_arrange_mode(self):
        """
        Turn on arrange mode, so each left-click on a box will be recorded in a sequence.
        We also tell the ImageLabel to display numbering.
        """
        #print("Enable arrange")
        self.in_arrange_mode = True
        self.arrange_order.clear()
        self.image_label.in_arrange_mode = True  # let the label draw numbers
        self.image_label.arrange_order = self.arrange_order
        self.image_label.update()

        for btn in self.all_buttons:
          btn.setStyleSheet("opacity: 0.5;")
          btn.setEnabled(False)
          
        self.noarrange_button_shortcut.setEnabled(True)
        self.arrange_mode_off_button.setStyleSheet("")
        self.arrange_mode_off_button.setEnabled(True)
        self.arrange_button_shortcut.setEnabled(False)

    def disable_arrange_mode(self):
        """
        Turn off arrange mode, then reorder bounding boxes/text 
        according to the recorded sequence, and reset everything.
        """
        #print("Disbale arrange")
        self.in_arrange_mode = False
        self.image_label.in_arrange_mode = False

        # Actually reorder bounding boxes/text in the order clicked
        self.reorder_boxes_and_text_by_click_order()

        # Clear the arrange order
        self.arrange_order.clear()
        self.image_label.arrange_order = self.arrange_order
        self.image_label.update()

        for btn in self.all_buttons:
          btn.setStyleSheet("")
          btn.setEnabled(True)
          
        self.arrange_button_shortcut.setEnabled(True)
        self.arrange_mode_off_button.setEnabled(False)
        self.arrange_mode_off_button.setStyleSheet("opacity: 0.5;")
        self.noarrange_button_shortcut.setEnabled(False)

    def arrange_bounding_box_sequence(self, box_dicts):
      """
      Takes a list of bounding boxes (each is a dict with 'id' and 'coords')
      and returns a new list in the desired order.
      
      For now, it's just a placeholder that returns the list unchanged.
      """
      # TODO: implement your custom reordering logic in the future.
      return box_dicts
    
    def on_arrange_button(self):
      """
      When the user clicks the "Arrange Boxes/Text" button,
      gather all bounding box + text data, reorder them,
      then re-populate the UI in the new order.
      """
      # 1) Gather current data from UI
      file_data = self.gather_file_data_from_ui()
      # file_data is like [ { "id":..., "coords":(x,y,w,h), "lines":[...] }, ...]

      # 2) Reorder them using our placeholder function
      new_data = self.arrange_file_data(file_data)
      # for now, arrange_file_data will just return the same sequence

      # 3) Update the image_label.bounding_boxes
      self.image_label.bounding_boxes = [
          {"id": d["id"], "coords": d["coords"]}
          for d in new_data
      ]

      # 4) Re-populate the text_list
      self.text_list.clear()
      for box_info in new_data:
          for line in box_info["lines"]:
              item = QListWidgetItem(line)
              item.setData(Qt.ItemDataRole.UserRole, box_info["id"])
              item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
              self.text_list.addItem(item)

      # 5) Update the display if needed
      self.image_label.update()

    def arrange_file_data(self, file_data):
      """
      Placeholder: returns the list unchanged. 
      In the future, you can add custom logic to sort or arrange them.
      """
      # This should be same as 
      # arrange_bouding_box_sequence

      return file_data[::-1]
    def reorder_boxes_and_text_by_click_order(self):
      """
      Reorder bounding boxes/text so that:
        - Boxes in self.arrange_order come first, in the order they appear in that list.
        - Any leftover boxes remain in their original order afterward.
      """
      # 1) Gather current data from UI
      file_data = self.gather_file_data_from_ui()
      # file_data is a list of dicts like:
      #   [ { "id": box_id, "coords":(x, y, w, h), "lines":[...] }, ... ]
  
      # 2) Create a quick lookup for each box_id
      box_id_map = { d["id"]: d for d in file_data }
  
      # 3) Identify which boxes were clicked
      used_ids = set(self.arrange_order)
  
      # 4) leftover = the boxes not in arrange_order, in original order
      leftover = [d for d in file_data if d["id"] not in used_ids]
  
      # 5) Build new_data: first the clicked ones in the clicked order,
      #    then the leftover in the original order
      new_data = []
      for bid in self.arrange_order:
          if bid in box_id_map:
              new_data.append(box_id_map[bid])
      new_data.extend(leftover)
  
      # 6) Update the image_label.bounding_boxes with the new order
      self.image_label.bounding_boxes.clear()
      for d in new_data:
          self.image_label.bounding_boxes.append({
              "id": d["id"],
              "coords": d["coords"]
          })
  
      # 7) Re-populate self.text_list and user_text_list in the same new order
      self.text_list.clear()
      self.user_text_list.clear()
      for d in new_data:
          for line in d["lines"]:
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, d["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.text_list.addItem(item)
          for user_line in d.get("user_lines", []):
            item2 = QListWidgetItem(user_line)
            item2.setData(Qt.ItemDataRole.UserRole, d["id"])
            item2.setFlags(item2.flags() | Qt.ItemFlag.ItemIsEditable)
            self.user_text_list.addItem(item2)
              
      # 8) Save it back to self.boxes_data for this image
      file_path = self.image_files[self.current_image_index]
      self.boxes_data[file_path] = new_data
  
      self.image_label.update()

    ########################################################################
    # Cleanup
    ########################################################################
    def export_raw_text(self):
      """
      Creates a file called RAW.txt in the current image directory.
      For each image, writes:
        Page <N>
        (all text lines)
      in order.
      """
      # 1) Make sure we have a valid directory
      if not self.image_directory:
          print("No directory opened yet.")
          return
  
      raw_path = os.path.join(self.image_directory, "RAW.txt")
  
      # 2) Open the file for writing (overwrite each time)
      with open(raw_path, "w", encoding="utf-8") as f:
          # 3) Enumerate all images in order
          for idx, file_path in enumerate(self.image_files, start=1):
              # Write "Page N"
              f.write(f"Page {idx}\n")
  
              # 4) Retrieve box data from self.boxes_data
              #    Example structure: 
              #    self.boxes_data[file_path] = [
              #      { "id": ..., "coords": (x,y,w,h), "lines": [ ... ] }, ...
              #    ]
              box_list = self.boxes_data.get(file_path, [])
              # For each bounding box, write all lines
              for box_info in box_list:
                  lines = box_info.get("lines", [])
                  for line in lines:
                      f.write(line + "\n")
  
              # Extra blank line between pages
              f.write("\n")
  
      self.log(f"Saved to: {raw_path}")

    def export_user_text(self):
        """
        Creates/overwrites TL.txt in the current directory.
        For each image, writes:
        Page <N>
        (all user lines for that image)
        """
        if not self.image_directory:
            print("No directory opened, cannot export user text.")
            return

        tl_path = os.path.join(self.image_directory, "TL.txt")

        # Make sure we have the latest edits from the current image, if needed
        self.update_in_memory_annotations()
        self.save_current_annotations()  # so everything is in the JSON as well

        with open(tl_path, "w", encoding="utf-8") as f:
            for idx, file_path in enumerate(self.image_files, start=1):
                f.write(f"Page {idx}\n")
                box_list = self.boxes_data.get(file_path, [])

                # For each bounding box, we expect a "user_lines" array
                for box_info in box_list:
                    user_lines = box_info.get("user_lines", [])
                    for line in user_lines:
                        f.write(line + "\n")

                f.write("\n")  # blank line between pages

        self.log(f"Saved to: {tl_path}")

    def closeEvent(self, event):
        self.save_current_annotations()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
