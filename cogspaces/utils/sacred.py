from sacred.observers import FileStorageObserver


class OurFileStorageObserver(FileStorageObserver):
    def artifact_event(self, name, filename):
        self.run_entry['artifacts'].append(name)
        self.save_json(self.run_entry, 'run.json')