import os
import uuid
from datetime import datetime


class WorkspaceService:
    def create_task_workspace(self, base_dir: str = "./workspaces") -> tuple[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:6]
        task_id = f"task_{timestamp}_{short_id}"
        workspace_dir = os.path.join(base_dir, task_id)
        os.makedirs(workspace_dir, exist_ok=True)
        return task_id, os.path.abspath(workspace_dir)
