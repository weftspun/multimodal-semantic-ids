# Decode binary Godot resources to text, in-place, using a SceneTree as the headless harness.
# Run per project:  godot --headless --path <project> --script <this>.gd
# Re-serialization only: ResourceLoader.load(bin) -> ResourceSaver.save(res, text_path).
# .scn -> .tscn, .res -> .tres. NO scene instantiation, NO node-tree dump.
extends SceneTree


func _initialize() -> void:
	var count := 0
	var fail := 0
	for path: String in _scan("res://"):
		var res := ResourceLoader.load(path)
		if res == null:
			printerr("LOAD_FAIL ", path)
			fail += 1
			continue
		var ext: String = path.get_extension().to_lower()
		var out: String = path.get_basename() + ("." + ("tscn" if ext == "scn" else "tres"))
		var err := ResourceSaver.save(res, out)
		if err == OK:
			print("OK ", path, " -> ", out)
			count += 1
		else:
			printerr("SAVE_FAIL ", out, " err=", err)
			fail += 1
	print("DECODED=", count, " FAILED=", fail)
	quit()


func _scan(root: String) -> Array[String]:
	var found: Array[String] = []
	var dir := DirAccess.open(root)
	if dir == null:
		return found
	dir.list_dir_begin()
	var name := dir.get_next()
	while name != "":
		if not name.begins_with("."):
			var full: String = root.path_join(name)
			if dir.current_is_dir():
				found.append_array(_scan(full))
			else:
				var e: String = name.get_extension().to_lower()
				if e == "scn" or e == "res":
					found.append(full)
		name = dir.get_next()
	dir.list_dir_end()
	return found
